"""
增量法SFM, 核心步骤在于视图对寻找（SLAM常为顺序列）、特征点匹配、本质矩阵计算、几何一致性检测、本质矩阵分解、PnP计算位姿、点云融合、BA捆绑调整等。
世界坐标默认为视图1的相机坐标。
由于基础矩阵分解时尺度是齐次的，因此所有视图的尺度与初始视图0、视图1分解E的平移向量t的尺度一致。这也间接说明了我们重建得到的是一个相对三维空间位置。
sfm基本步骤：
1. 将所有图像downscale并提取所有图像的sift 特征，得到特征点和描述子。
2. 初始化sfm：计算视图0和某一视图的本质矩阵，一致性检测去除噪点，分解得到T和R并重建为初始化点云，视图0的相机坐标系为世界坐标系；
3. 不断增加视图0,1,2,...i,i+1.. 视图i+1与视图i进行匹配，选择i-1和i匹配与i和i+1视图匹配的公共已重建三维点计算PnP视图i+1的位姿，然后fusion新增点云；
4. 每一步骤都要进行bundle adjustment。
"""
import collections
import os
import sys
import time
import utils.utils
import tqdm
import logging

import cv2
import numpy as np
from scipy.optimize import least_squares
from tqdm.contrib.logging import logging_redirect_tqdm


class IncrementalSFM:
    def __init__(self, imgs_dir: str, downscale_factor: float = 1.0, log_handler=None):
        """
        初始化整个sfm的相关参数、中间变量，并读取图像，进行下采样处理。

        :param imgs_dir:
        :param downscale_factor:
        :param log_handler:
        """
        # set log handler, 用于将log输出到文件
        self.log_handler = None
        if log_handler is not None:
            self.log_handler = log_handler
            logging.getLogger().addHandler(log_handler)
        # read images
        img_names = os.listdir(imgs_dir)
        self.imgs_dir = imgs_dir
        self.downscale_factor = downscale_factor # downscale以加快重建速度
        self.img_list = [] # 存储所有图像
        self.img_name_list = [] # 存储所有图像的名称
        for name in img_names:
            if name[-4:] == '.txt':
                self.K = np.loadtxt(imgs_dir + '/' + name, dtype=str).astype(float)
                # downscale以加快重建速度
                self.K[0, 0] /= self.downscale_factor
                self.K[1, 1] /= self.downscale_factor
                self.K[0, 2] /= self.downscale_factor
                self.K[1, 2] /= self.downscale_factor
            else:
                # downscale the img
                img = utils.utils.downscale_img(cv2.imread(imgs_dir + '/' + name), self.downscale_factor)
                self.img_list.append(img)
                self.img_name_list.append(name.split('.')[0])
        # 待重建的位姿和Points
        self.R0 = np.array([[1.0, 0, 0], [0, 1.0, 0], [0, 0, 1.0]])  # 视图0的位姿
        self.T0 = np.zeros((3, 1), dtype=float)
        self.Rs = [] # 重建的位姿R
        self.Ts = [] # 重建的位姿T
        self.Points = np.zeros((1, 3), dtype=float) # 重建的三维点
        self.Colors = np.zeros((1, 3), dtype=np.uint8) # 重建的三维点color
        # (p,u,v)三维点由第p个视图重建得到,像素坐标为(u,v)。方便计算重投影误差或者快速索引进行局部/全局的BA优化
        self.pts_info = np.zeros((1, 3), dtype=float)

        self.order = []  # 记录视图加入的顺序索引,不计入视图0。order[i]第i个视图的索引(即img_list[order[i]])
        self.reconstruct_num = []  # 记录每个视图重建的点数
        self.kps_list = []  # sift特征点
        self.des_list = []  # sift特征描述子

        # 这是为了1. 增量时快速找到连续视图匹配的重合关键点以使用PnP计算位姿; 2.过滤重合关键点重建三维坐标。
        self.h = []  # 同order顺序, self.h[i][j]记录第i个视图sift提取的第j个关键点所重建得到的Points下标。j为该视图关键点的trainIdx。

        # book[i]表示第i个视图已经添加过了, 防止重复添加
        self.book = collections.defaultdict(int)

        # sfm状态:
        # -1:待机
        # 0: 完成提取特征
        # 1: 完成初始化sfm+进行增量重建
        # 2: 完成step增量重建
        # 3: 完成select和全局BA, over
        self.state = -1
        self.img_matches = None  # 临时存储匹配的matches, 用于pyqt显示
        self.new_points = None  # 临时存储新重建的点, 用于pyqt显示
        if self.log_handler is not None:
            logging.info('Initialize done, read {} images.'.format(len(self.img_list)))

    def get_nxt_view(self, match_factor=0.7):
        """
        获得下一个视图,返回上一个视图和该视图匹配像素点和匹配的matches

        :param match_factor: 次优过滤因子
        :return: P_pix1, P_pix2, matches
        """
        # i:上一个视图的下标
        if len(self.order) == 0:
            self.book[0] = 1
            i = 0
        else:
            i = self.order[-1]
        kps1, des1 = self.kps_list[i], self.des_list[i]
        # 顺序视图
        for j in range(i + 1, len(self.img_list)):
            if self.book[j]:
                continue
            self.book[j] = 1  # 该视图已被尝试fussion
            kps2, des2 = self.kps_list[j], self.des_list[j]
            # 用flann的kd_tree来匹配特征点,不能保证最优解。也可使用暴力匹配BFMMatch能保证最优解复杂度高。
            # FLANN_INDEX_KDTREE = 0
            # index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
            # search_params = dict(checks=50)
            # flann = cv2.FlannBasedMatcher(index_params, search_params)
            bf = cv2.BFMatcher()
            matches = bf.knnMatch(des1, des2, k=2)
            # 用次优解过滤
            good = []
            for match in matches:
                if match[0].distance < match_factor * match[1].distance:
                    good.append(match[0])
            if len(good) < 8:
                print('Too few matches between {} and {}'.format(self.img_name_list[i], self.img_name_list[j]))
                if self.log_handler:
                    logging.info(
                        'Too few matches between {} and {}'.format(self.img_name_list[i], self.img_name_list[j]))
                continue
            self.order.append(j)
            matches = np.array(good)
            return np.array([kps1[x.queryIdx].pt for x in matches]), np.array(
                [kps2[x.trainIdx].pt for x in matches]), matches

        return None, None, None

    def extract_feat(self, name='sift'):
        """
        提取所有img_list的特征点和描述子

        :param name:
        :return:
        """
        extractor = cv2.xfeatures2d.SIFT_create()
        print("extract {} features...".format(name))
        if self.log_handler:
            logging.info("extract {} features...".format(name))
        with logging_redirect_tqdm():
            for i in tqdm.tqdm(range(len(self.img_list)), file=sys.stdout):
                # gray the img
                img = cv2.cvtColor(self.img_list[i], cv2.COLOR_BGR2GRAY)
                kps, des = extractor.detectAndCompute(img, None)
                tqdm.tqdm.write('%s extract %d key points' % (self.img_name_list[i], len(kps)))
                # 将进度条写入日志
                if self.log_handler:
                    logging.info('%s extract %d key points' % (self.img_name_list[i], len(kps)))
                self.kps_list.append(kps)
                self.des_list.append(des)
        print("extract {} features done".format(name))
        if self.log_handler:
            logging.info("extract {} features done".format(name))

    def initialize_sfm(self, if_local_BA=False, match_factor=0.7, threshold=0.4):
        """
        初始sfm选择两个视图进行双目重建, 基准视图为视图0。该步骤很重要,基本直接决定了后续的重建效果。

        :param if_local_BA: 是否进行局部BA
        :param match_factor: 次优过滤因子
        :param threshold: 本质矩阵RANSAC阈值
        :return:
        """
        print("initialize sfm...")
        if self.log_handler:
            logging.info("initialize sfm...")
        P_pix1, P_pix2, matches = self.get_nxt_view(match_factor=match_factor)
        # 基于8点法和RANSAC计算本质矩阵, 并利用本质矩阵进行几何一致性检测得到mask。
        E, mask = cv2.findEssentialMat(P_pix1, P_pix2, self.K, method=cv2.RANSAC, prob=0.999, threshold=threshold)

        # 分解本质矩阵E=t^R ：svd分解得到4个解, 并根据深度>0进行验证。还可以通过重投影进行一致性检测。
        _, R, T, mask = cv2.recoverPose(E, P_pix1, P_pix2, self.K, mask=mask)
        keep = np.where(mask > 0)[0]
        P_pix1, P_pix2, matches = P_pix1[keep], P_pix2[keep], matches[keep]

        # 特征匹配可视化
        self.img_matches = cv2.drawMatches(self.img_list[0], self.kps_list[0], self.img_list[1], self.kps_list[1],
                                           matches,
                                           None, flags=cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)

        # 添加该视图的位姿
        self.Rs.append(R)
        self.Ts.append(T)
        # 三角测量得到初始重建点云
        P_w = utils.utils.triangulation(self.R0, self.T0, self.Rs[-1], self.Ts[-1], P_pix1, P_pix2, self.K)
        # 计算重投影误差
        err = utils.utils.reprojection_err(self.K, P_pix2, P_w, R, T)
        print("initial sfm with view %s and %s, Reprojection Err: %.3f, " % (
            self.img_name_list[0], self.img_name_list[self.order[0]], err), end='')
        if self.log_handler:
            logging.info("initial sfm with view %s and %s, Reprojection Err: %.3f, " % (
                self.img_name_list[0], self.img_name_list[self.order[0]], err))
        if if_local_BA:
            ti=time.time()
            P_w, R, T = self.local_bundle_adjustment(P_pix2, P_w, self.K, R, T)
            err = utils.utils.reprojection_err(self.K, P_pix2, P_w, R, T)
            print("BA Reprojection Err: %.3f, cost time: %.3fs" % (err,time.time()-ti), end='')
            if self.log_handler:
                logging.info("BA Reprojection Err: %.3f, cost time: %.3fs" % (err,time.time()-ti))
        print('')
        # update Points and Colors and pts_info
        self.Points = np.vstack([self.Points, P_w])
        self.new_points = P_w  # 用于pyqt5显示的新加入的点云
        self.reconstruct_num.append(len(P_w) + 1)
        color = np.array([self.img_list[self.order[-1]][int(x[1])][int(x[0])] for x in P_pix2])
        pts_info = np.concatenate([np.ones((len(P_pix2), 1)) * (len(self.order) - 1), P_pix2], axis=1)
        self.Colors = np.vstack([self.Colors, color])
        self.pts_info = np.vstack([self.pts_info, pts_info])

        # update h
        d = {match.trainIdx: i + 1 for i, match in enumerate(matches)}
        self.h.append(d)

        print("initialize sfm done")
        if self.log_handler:
            logging.info("initialize sfm done")

    def get_common_points(self, matches, P_pix_cur):
        """
        当前新加入的视图c与上一个视图b的匹配为matches,b与a的匹配为self.h[-1]
        寻找公共点,并返回公共点的3D坐标和2D像素坐标

        :param matches: 当前视图c与上一个视图b的匹配
        :param P_pix_cur: 当前视图c的像素坐标
        :return:
        """
        map_func = np.vectorize(lambda match: self.h[-1].get(match.queryIdx, -1))
        pts_idx = map_func(matches)
        cm = np.where(pts_idx > 0)[0]  # 公共点
        return self.Points[pts_idx[cm]], P_pix_cur[cm]

    def fusion_cur_view(self, Points, matches, P_pix_cur):
        """
        融合点云,重合的匹配点不融合,其他点融合到self.Points内并维护更新self.h

        :param Points: 当前视图所有match三角化的3D坐标
        :param matches: 当前视图与上一个视图的匹配
        :param P_pix_cur: 当前视图的像素坐标
        :return:
        """
        map_func = np.vectorize(lambda match: self.h[-1].get(match.queryIdx, -1))
        pts_idx = map_func(matches)
        new_idx = np.where(pts_idx == -1)[0]  # 非公共点需被融合
        cm_idx = np.where(pts_idx > 0)[0]  # 上个视图已重建的三维点及公共点不被融合

        # 非公共点需要被融合, 因此h映射到新的点云下标
        d = {matches[idx].trainIdx: i + len(self.Points) for i, idx in enumerate(new_idx)}
        # 公共点不需要被融合,h映射到之前的下标。
        [d.__setitem__(matches[i].trainIdx, pts_idx[i]) for i in cm_idx]
        self.h.append(d)  # 更新h
        # 更新self.Points、Colors、pts_info等
        self.Points = np.vstack([self.Points, Points[new_idx]])
        self.reconstruct_num.append(len(Points[new_idx]))
        cur_img = self.img_list[self.order[-1]]
        color = np.array([cur_img[int(P_pix_cur[i][1])][int(P_pix_cur[i][0])] for i in new_idx])
        self.Colors = np.vstack([self.Colors, color])
        pts_info = np.concatenate([np.ones((len(new_idx), 1)) * (len(self.Rs) - 1), P_pix_cur[new_idx]], axis=1)
        self.pts_info = np.vstack([self.pts_info, pts_info])
        return Points[new_idx], P_pix_cur[new_idx]

    def local_bundle_adjustment(self, x, X, K, R, T):
        """
        局部光束法平差,针对一个位姿和其重建点进行优化, 优化变量为位姿和重建点
        目标函数: 将重投影误差(重投影坐标差)设置为残差, 最小化残差平方和, 默认调用简单的牛顿法优化

        :param x: 像素坐标
        :param X: 三维坐标
        :param K: 内参
        :param R: 位姿
        :param T: 位姿
        :return:  BA后的三维点和位姿
        """

        def func(variables, K, x):
            """
            目标函数

            :param variables: R, T, X
            :param K: 内参
            :param x: 像素坐标
            :return:
            """
            R = variables[:9].reshape(3, 3)
            T = variables[9:12].reshape(3, 1)
            X = variables[12:].reshape(-1, 3)

            R_vector, _ = cv2.Rodrigues(R)
            reproj_x, _ = cv2.projectPoints(X, R_vector, T, K, distCoeffs=np.zeros(5))
            reproj_x = np.squeeze(reproj_x)
            return x.ravel() - reproj_x.ravel()

        variables = np.hstack([R.ravel(), T.ravel(), X.ravel()])
        BA_vars = least_squares(func, variables, args=(K, x)).x
        R = BA_vars[:9].reshape(3, 3)
        T = BA_vars[9:12].reshape(3, 1)
        X = BA_vars[12:].reshape(-1, 3)
        return X, R, T

    def global_bundle_adjustment(self):
        """
        全局光束法平差,最小化所有位姿、重建点的重投影误差, 优化变量为R、T、X, 优化方法为LM
        输入参数包括: 相机内参K, 所有视图的位姿R、T, 所有视图的像素坐标P_pix, 所有视图的三维点P_w
        其中各个点的下标信息存储在pts_info中,pts_info的第一列为视图下标,第二、三列为像素坐标

        :return:
        """

        def func(variables, K, pts_info, n):
            """
            全局光束法平差的目标函数

            :param variables: 优化变量,包括所有视图的位姿R、T, 所有视图的三维点P_w
            :param K: 相机内参
            :param pts_info: 所有三维点的信息: (i, u, v),i为视图下标表示当前点由self.Rs[i]即视图self.order[i]重建得到, u、v为像素坐标
            :param points_num:
            :return:
            """
            # variables: [R1, T1, ..., Rn, Tn, X1, ..., Xn]
            R = variables[:9 * n].reshape(n, 3, 3)
            T = variables[9 * n:12 * n].reshape(n, 3, 1)
            X = variables[12 * n:].reshape(-1, 3)

            err = np.zeros(shape=(1, 2))

            for i in range(n):
                pos = np.where(pts_info[:, 0] == i)[0]  # 第i个视图重建得到的所有点
                P_pix = pts_info[pos][:, 1:]  # 第i个视图重建得到的所有点的像素坐标
                P_w = X[pos]  # 第i个视图重建得到的所有点的三维坐标
                reproj = np.matmul(K, np.matmul(R[i], P_w.T) + T[i]).T  # 第i个视图重建得到的所有点的重投影坐标
                reproj = reproj[:, :2] / reproj[:, 2:]  # 第i个视图重建得到的所有点的重投影坐标
                err = np.vstack([err, P_pix - reproj])  # 第i个视图重建得到的所有点的重投影误差
            return err.ravel()

        n = len(self.Rs)
        # BA优化variables: [R1, R2, ..., Rn, T1, T2, ..., Tn, X1, ..., Xn]
        variables = np.hstack([R.ravel() for R in self.Rs] + [T.ravel() for T in self.Ts] + [self.Points.ravel()])
        # 计算BA前的重投影误差
        # 待修改............................................................ 每次func copy pts_info, 速度慢
        err = func(variables, self.K, self.pts_info, n)
        err = err.reshape(-1, 2)
        reprojection_error = np.sqrt(np.sum(err ** 2, axis=1)).mean()
        print('Before global BA, reprojection error: ', reprojection_error)
        # 开始优化
        BA_vars = least_squares(func, variables, args=(self.K, self.pts_info, n))
        variables = BA_vars.x

        # 更新self.Rs、self.Ts、self.Points
        self.Rs = [variables[i * 9:(i + 1) * 9].reshape(3, 3) for i in range(n)]
        self.Ts = [variables[9 * n + i * 3:9 * n + (i + 1) * 3].reshape(3, 1) for i in range(n)]
        self.Points = variables[12 * n:].reshape(-1, 3)

        # 计算BA后的重投影误差
        err = func(variables, self.K, self.pts_info, n)
        err = err.reshape(-1, 2)
        reprojection_error = np.sqrt(np.sum(err ** 2, axis=1)).mean()
        print('After global BA, reprojection error: ', reprojection_error)

    def increment(self, if_local_BA=False, match_factor=0.7, threshold=0.4):
        """
        增量过程, 即state=1, 每步get_nxt_view()得到下一视图, 并进行几何一致性检测
        获取当前视图已重建得到的三维点计算PnP得到位姿, 再三角化, 点云融合, BA优化...

        :param if_local_BA: 是否进行局部BA
        :param match_factor: 次优过滤阈值
        :param threshold: ransac阈值
        :return:
        """
        print('Incrementing ...')
        if self.log_handler is not None:
            logging.info('Incrementing ...')
        ti = time.time()
        # 目前是顺序的视图添加增量法
        P_pix1, P_pix2, matches = self.get_nxt_view(match_factor=match_factor)
        # over, 视图添加完毕
        if P_pix1 is None:
            return False
        # 利用本质矩阵E和基础矩阵F进行几何一致性检测
        # _, mask = cv2.findFundamentalMat(P_pix1, P_pix2, cv2.FM_RANSAC, ransacReprojThreshold=0.4, confidence=0.999,
        #                                  maxIters=None)
        _, mask = cv2.findEssentialMat(P_pix1, P_pix2, self.K, method=cv2.RANSAC, prob=0.999, threshold=threshold)
        keep = np.where(mask > 0)[0]
        P_pix1, P_pix2, matches = P_pix1[keep], P_pix2[keep], matches[keep]
        # 特征匹配点可视化
        self.img_matches = cv2.drawMatches(self.img_list[self.order[-2]], self.kps_list[self.order[-2]],
                                           self.img_list[self.order[-1]], self.kps_list[self.order[-1]], matches, None,
                                           flags=cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)

        # 获得与上一个视图的公共匹配点, 利用公共匹配点计算PnP位姿。
        obj_points, img_points = self.get_common_points(matches, P_pix2)
        # 公共匹配点过少, 不足以计算稳定、精确的位姿
        if len(obj_points) < 6:
            print('Too few common points, drop view %s' % (self.img_name_list[self.order[-1]]))
            if self.log_handler is not None:
                logging.info('Too few common points, drop view %s' % (self.img_name_list[self.order[-1]]))
            self.order.remove(self.order[-1])

            return True
        # 位姿计算增量时使用PnP算法,并非视图0和当前视图进行匹配。PnP的输入点要求是上一个视图匹配的重合点。
        # PnP计算位姿: 已知3D点、像素点和K求解位姿R、T，可以直接解线性方程/LM等迭代算法/P3P等方法。
        _, R_vector, T, keep = cv2.solvePnPRansac(obj_points, img_points, self.K, distCoeffs=np.zeros(5),
                                                  flags=cv2.SOLVEPNP_ITERATIVE)
        R, _ = cv2.Rodrigues(R_vector)
        self.Rs.append(R)
        self.Ts.append(T)
        # 三角测量重建当前视图的坐标
        Points = utils.utils.triangulation(self.Rs[-2], self.Ts[-2], R, T, P_pix1, P_pix2, self.K)
        # 融合当前视图点云到self.Points并更新h、pts_info、Colors等。
        Points, P_pix2 = self.fusion_cur_view(Points, matches, P_pix2)
        # 计算重投影误差
        err = utils.utils.reprojection_err(self.K, P_pix2, Points, R, T)
        print("add view %s with %d Points, Reprojection err:%3f, " % (
            self.img_name_list[self.order[-1]], len(Points), err), end='')
        if self.log_handler:
            logging.info("add view %s with %d Points, Reprojection err:%3f, " % (
                self.img_name_list[self.order[-1]], len(Points), err))
        # BA视图的位姿、三维点
        if if_local_BA:
            ti= time.time()
            Points, R, T = self.local_bundle_adjustment(P_pix2, Points, self.K, R, T)
            self.Points[-len(Points):], self.Rs[-1], self.Ts[-1] = Points, R, T
            # 计算BA后的重投影误差
            err = utils.utils.reprojection_err(self.K, P_pix2, Points, R, T)
            print("BA Reprojection err:%3f, cost time: %.3fs" % (err,time.time()-ti), end='')
            if self.log_handler:
                logging.info("BA Reprojection err:%3f, cost time: %.3fs" % (err,time.time()-ti))
        print('')
        self.new_points = self.Points[-len(Points):]
        # print("fussion img {} cost time {}s".format(self.order[-1], time.time() - ti))
        return True

    def select_global_points(self, threshold=0.5):
        """
        剔除一些重投影误差明显很大的点, 即剔除一些outlier, 保留inlier
        稀疏重建中, 一般认为重投影误差小于0.5的点为inlier, 大于0.5的点为outlier
        处理的数据: self.Points, self.Colors, self.pts_info

        :param threshold: 保留inlier的阈值
        :return:
        """
        n = len(self.Rs)  # 视图数量
        keep_idx = np.zeros(len(self.Points), dtype=bool)
        for i in range(n):  # 遍历每个视图
            R, T = self.Rs[i], self.Ts[i]
            pos = np.where(self.pts_info[:, 0] == i)[0]  # 第i个视图的点云索引
            P_w = self.Points[pos]
            P_pix = self.pts_info[pos, 1:3]
            # 计算重投影误差
            reproj_x = (self.K @ (R @ P_w.T + T)).T
            reproj_x = reproj_x[:, :2] / reproj_x[:, 2:]
            err = np.linalg.norm(reproj_x - P_pix, axis=1)
            # 保留重投影误差小于阈值的点
            keep_idx[pos[np.where(err < threshold)[0]]] = True
        # 更新数据
        print("before select: {} points, select {} inlier points".format(len(self.Points), np.sum(keep_idx)))
        if self.log_handler:
            logging.info("before select: {} points, select {} inlier points".format(len(self.Points), np.sum(keep_idx)))
        self.Points, self.Colors, self.pts_info = self.Points[keep_idx], self.Colors[keep_idx], self.pts_info[keep_idx]

    def step(self, if_local_BA=False, if_global_BA=False, if_select=True, match_factor=0.7, threshold=0.4):
        """
        sfm的过程被我抽象成一个状态机, 每个step就是一个状态, 每个状态都有对应的任务目标
        sfm状态:
        -1:待机
        0: 完成特征提取
        1: 完成初始化sfm+增量重建ing
        2: 完成step增量重建
        3: 完成select和全局BA,over
        
        :param if_local_BA: 局部BA, 优点: 速度快, 缺点: 通常情况下鲁棒性强, 但某些视图容易发生偏移
        :param if_global_BA: 全局BA, 优点: 精度高, 缺点: 计算量大, 速度慢
        :param if_select: 是否剔除outlier, 阈值固定为0.5
        :param match_factor: sift的最优、次优匹配点距离比例, 分两种情况:1. 图片分辨率高, 特征点数量多, 此时需剔除一些错误关键点, 即设置较小的match_factor; 2. 图片分辨率低, 特征点数量少, 此时需保留更多的关键点用以RANSAC, 即设置较大的match_factor
        :param threshold: ransac内点阈值
        :return:
        """

        if self.state == -1:  # 待机状态: 提取特征
            self.extract_feat()
            self.state += 1
        elif self.state == 0:  # 提取特征: 初始化sfm
            # 初始化sfm
            self.initialize_sfm(if_local_BA=if_local_BA, match_factor=match_factor, threshold=threshold)
            self.state += 1
        elif self.state == 1:  # 初始化sfm: 增量sfm
            # 增量sfm
            if_continue = self.increment(if_local_BA=if_local_BA, match_factor=match_factor, threshold=threshold)
            if not if_continue:  # 增量sfm over
                print("all images have been reconstructed!")
                if self.log_handler:
                    logging.info("all images have been reconstructed!")
                self.state += 1
        elif self.state == 2:  # 完成增量sfm->全局BA
            if if_select:  # select
                print("select global points...")
                if self.log_handler:
                    logging.info("select global points...")
                self.select_global_points(threshold=0.5)
            if if_global_BA:  # 全局BA
                print("global BA...")
                if self.log_handler:
                    logging.info("global BA...")
                ti = time.time()
                self.global_bundle_adjustment()
                print("global BA completed! len(Points):%d, cost time:%.3f" % (len(self.Points), time.time() - ti))
                if self.log_handler:
                    logging.info(
                        "global BA completed! len(Points):%d, cost time:%.3f" % (len(self.Points), time.time() - ti))
            self.state += 1  # over

    def save(self, path):
        """
        将重建得到的self.Points和self.Colors保存为ply文件

        :param path:
        :return:
        """
        # 'result/' + self.imgs_dir.split('/')[-1] + '.ply'
        utils.utils.save_as_ply(self.Points, self.Colors, path)

    def check_finished(self):
        """
        检查是否所有视图完成重建

        :return:
        """
        if len(self.book) == len(self.img_name_list):
            return True
        return False

    def check_sfm_completed(self):
        """
        检查sfm是否完成, 指标: self.state==3即select和全局BA完成

        :return:
        """
        if self.state == 3:
            return True
        return False

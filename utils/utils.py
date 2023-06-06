import cv2
import numpy as np


def downscale_img(img, downscale_factor):
    if downscale_factor == 1:
        return img
    return cv2.resize(img, (int(img.shape[1] // downscale_factor), int(img.shape[0] // downscale_factor)),
                      cv2.INTER_LINEAR)


def triangulation(R1, T1, R2, T2, P_pix1, P_pix2, K):
    """
    三角测量
    s1x1=KM1X, s2x2=KM2X
    其中x1、x2是像素坐标的齐次形式,X是三维坐标的齐次形式, M1=[R1,T1],M2=[R2,T2],X=[X,1]^T
    两个式子分别左乘x1^和x2^可以得到一个线性方程组。

    """
    pose_1 = np.matmul(K, np.hstack([R1, T1]))
    pose_2 = np.matmul(K, np.hstack([R2, T2]))
    P_w_h = cv2.triangulatePoints(pose_1, pose_2, P_pix1.T, P_pix2.T)
    return (P_w_h / P_w_h[3])[:3].T


def save_as_ply(P, C, save_dir):
    header = '''ply
format ascii 1.0
element vertex {}
property float x
property float y
property float z
property uchar blue
property uchar green
property uchar red
end_header
'''
    mean_X = np.mean(P, axis=0)
    P -= mean_X
    obj = np.hstack([P, C])
    with open(save_dir, 'w') as f:
        f.write(header.format(len(obj)))
        np.savetxt(f, np.c_[obj], fmt="%f %f %f %d %d %d")


def reprojection_err(K, x, X, R, T):
    """
    功能: 已知三维坐标、像素坐标、内参K和位姿R、T, 计算重投影误差, sx=K(RX+T)

    :param K:
    :param x:
    :param X:
    :param R:
    :param T:
    :return:
    """
    M = np.hstack([R, T])
    X_hom = np.vstack([X.T, np.ones(len(X))])
    reproject_x_hom = np.matmul(K, np.matmul(M, X_hom))
    reproject_x = (reproject_x_hom / reproject_x_hom[-1])[:-1].T
    err = np.linalg.norm(reproject_x - x, axis=1).mean()
    return err

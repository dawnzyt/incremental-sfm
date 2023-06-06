from qtawesome import icon_browser
import qtawesome
from PyQt5 import Qt
from PyQt5.QtGui import *
from PyQt5 import QtCore
import piexif


def convert_icon(icon, size=None, fmt='JPG'):
    if size is None:
        size = icon.availableSizes()
    pixmap = icon.pixmap(size)
    array = QtCore.QByteArray()
    buffer = QtCore.QBuffer(array)
    buffer.open(QtCore.QIODevice.WriteOnly)
    pixmap.save(buffer, fmt)
    buffer.close()
    return array.data()


if __name__ == '__main__':
    # icon=qtawesome.icon('fa.meetup', color='white')
    # thumbnail = convert_icon(icon)
    # path='./'
    # exif = piexif.load(path)
    # exif['thumbnail'] = thumbnail
    # data = piexif.dump(exif)
    # piexif.insert(data, path)
    icon_browser.run()

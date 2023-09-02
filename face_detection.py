import sys
import cv2 as cv
import numpy as np
from skimage.measure import block_reduce as blrd
from skimage.metrics import structural_similarity as strsim
from scipy.spatial.distance import cosine as csn
from scipy.ndimage import gaussian_gradient_magnitude as ggm
from matplotlib import pyplot as plt
from matplotlib.widgets import Button as btn
from PyQt5 import QtCore, QtGui, QtWidgets

# класс интерфейса
class Ui_Interface(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(232, 183)
        MainWindow.setMinimumSize(QtCore.QSize(232, 183))
        MainWindow.setMaximumSize(QtCore.QSize(232, 183))
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(10, 10, 171, 21))
        font = QtGui.QFont()
        font.setPointSize(12)
        font.setBold(True)
        font.setWeight(75)
        self.label.setFont(font)
        self.label.setObjectName("label")
        self.label_2 = QtWidgets.QLabel(self.centralwidget)
        self.label_2.setGeometry(QtCore.QRect(10, 40, 151, 21))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.label_2.setFont(font)
        self.label_2.setObjectName("label_2")
        self.label_3 = QtWidgets.QLabel(self.centralwidget)
        self.label_3.setGeometry(QtCore.QRect(10, 70, 181, 21))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.label_3.setFont(font)
        self.label_3.setObjectName("label_3")
        self.label_4 = QtWidgets.QLabel(self.centralwidget)
        self.label_4.setGeometry(QtCore.QRect(10, 150, 181, 21))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.label_4.setFont(font)
        self.label_4.setObjectName("label_4")
        self.comboBox = QtWidgets.QComboBox(self.centralwidget)
        self.comboBox.setObjectName("comboBox")
        self.comboBox.setGeometry(QtCore.QRect(160, 150, 50, 21))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.comboBox.setFont(font)
        self.comboBox.setObjectName("comboBox")
        self.lineEdit = QtWidgets.QLineEdit(self.centralwidget)
        self.lineEdit.setGeometry(QtCore.QRect(160, 40, 31, 21))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.lineEdit.setFont(font)
        self.lineEdit.setText("")
        self.lineEdit.setObjectName("lineEdit")
        self.lineEdit_2 = QtWidgets.QLineEdit(self.centralwidget)
        self.lineEdit_2.setGeometry(QtCore.QRect(190, 70, 31, 21))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.lineEdit_2.setFont(font)
        self.lineEdit_2.setText("")
        self.lineEdit_2.setObjectName("lineEdit_2")
        self.pushButton = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton.setGeometry(QtCore.QRect(10, 110, 91, 31))
        self.pushButton.clicked.connect(self.button_click_1)
        font = QtGui.QFont()
        font.setPointSize(12)
        self.pushButton.setFont(font)
        self.pushButton.setObjectName("pushButton")
        self.pushButton_2 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_2.setGeometry(QtCore.QRect(130, 110, 91, 31))
        self.pushButton_2.clicked.connect(self.button_click_2)
        font = QtGui.QFont()
        font.setPointSize(12)
        self.pushButton_2.setFont(font)
        self.pushButton_2.setObjectName("pushButton_2")
        MainWindow.setCentralWidget(self.centralwidget)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "Программа"))
        self.label.setText(_translate("MainWindow", "Введите параметры:"))
        self.label_2.setText(_translate("MainWindow", "Кол-во эталонов ="))
        self.label_3.setText(_translate("MainWindow", "Кол-во изображений ="))
        self.label_4.setText(_translate("MainWindow", "База изображений:"))
        self.pushButton.setText(_translate("MainWindow", "Результат"))
        self.pushButton_2.setText(_translate("MainWindow", "Графики"))
        self.comboBox.addItem('1')
        self.comboBox.addItem('2')

    def button_click_1(self, Interface):
        Detection(int(self.lineEdit.text()), int(self.lineEdit_2.text())).show_results()

    def button_click_2(self, Interface):
        Detection(int(self.lineEdit.text()), int(self.lineEdit_2.text())).show_histograms()

# класс изображения
class Image():

    # инициализация изображения
    def __init__(self, i, j, tp):
        self.tp = tp
        db = ui.comboBox.currentText()

        if db == '1':
            self.image = cv.imread('ORL_Faces/s' + str(i + 1) + '/' + \
                                   str(j + 1) + '.pgm')

            self.image_gray = cv.imread('ORL_Faces/s' + str(i + 1) + '/' + \
                                        str(j + 1) + '.pgm', 0)

        elif db == '2':
            self.image = cv.imread('faces94/s' + str(i + 1) + '/' + \
                                   str(j + 1) + '.jpg')

            self.image_gray = cv.imread('faces94/s' + str(i + 1) + '/' + \
                                        str(j + 1) + '.jpg', 0)


    # возвращение изображения
    # нужного типа
    def get_image(self):
        if self.tp == 'image':
            return self.image

        elif self.tp == 'image_gray':
            return self.image_gray

        elif self.tp == 'image_float':
            return np.float32(self.image_gray)


# класс детекции
class Detection():

    # инициальзиция массива для эталона
    # n - кол-во эталонов
    # a переменная - начало массива классов
    # b переменная - конец массива классов
    # tp - тип метода 
    def __init__(self, n, m):
        self.n = n
        self.m = m
        db = ui.comboBox.currentText()

        if db == '1':
            self.r = 10
            self.t = self.r - self.n

        elif db == '2':
            self.r = 20
            self.t = self.r - self.n

    # выбор метода
    # x, y - индексация для
    # изображения
    def function(self, x, y, tp):
        if tp == 'bright_hist':
            image = Image(x, y, 'image_gray').get_image()

            # 512 - количество ячеек, [0, 256] - диапазон
            return cv.calcHist([image], [0], None, [512], [0, 256])

        elif tp == 'dft':
            image = Image(x, y, 'image_float').get_image()

            return cv.dft(image, flags=cv.DFT_COMPLEX_OUTPUT)

        elif tp == 'dct':
            image = Image(x, y, 'image_float').get_image()

            return cv.dct(image)

        elif tp == 'gradient':
            image = Image(x, y, 'image_float').get_image()

            return ggm(image, sigma=5)

        elif tp == 'scale':
            image = Image(x, y, 'image_gray').get_image()

            return blrd(image, (2, 1), np.max)

    # получение эталонных изображений
    def get_standards(self, tp):
        self.standards = [[] for x in range(self.m)]

        for i in range(self.m):
            for j in range(self.n):
                self.standards[i].append(self.function(i, j, tp))

    # сравнение эталонов
    # с тестовой выборкой
    # и вывод результата
    def compare_arrays(self, tp):
        result = [[] for x in range(self.m)]
        self.get_standards(tp)

        for i in range(self.m):
            for j in range(self.n, self.r):
                func = self.function(i, j, tp)
                classes = [[], []]

                for k in range(self.m):
                    compares = []

                    for l in range(self.n):

                        if tp == 'bright_hist':
                            compares.append(cv.compareHist(self.standards[k][l], func, cv.HISTCMP_CORREL) * 100)

                        elif tp == 'dft':
                            image = Image(i, j, 'image_float').get_image()

                            dist = np.linalg.norm(self.standards[k][l] - func)
                            max_dist = np.sqrt(image.shape[0] * image.shape[1] * 2) * 255

                            compares.append((max_dist / dist) * 100)

                        elif tp == 'dct':
                            similarity_score = 1 - csn(self.standards[k][l].flatten(), func.flatten())

                            compares.append(similarity_score * 100)

                        elif tp == 'gradient':
                            hist1 = cv.normalize(self.standards[k][l], self.standards[k][l], norm_type=cv.NORM_L1)
                            hist2 = cv.normalize(func, func, norm_type=cv.NORM_L1)

                            compares.append(cv.compareHist(hist1, hist2, cv.HISTCMP_CORREL) * 100)

                        elif tp == 'scale':
                            compares.append(strsim(self.standards[k][l], func, data_range=255) * 100)

                    classes[0].append(np.argmax(compares))
                    classes[1].append(np.max(compares))

                result[i].append([j, np.argmax(classes[1]), classes[0][np.argmax(classes[1])]])

        return result

    # получение результатов
    # работы программы для
    # каждого выбранного класса
    def get_results(self):
        results = []

        for method in 'bright_hist', 'dft', 'dct', 'gradient', 'scale':
            results.append(self.compare_arrays(method))

        return results

    def show_results(self):
        results = self.get_results()
        self.flag = True
        
        def stop(event):
            self.flag = not self.flag

        fig = plt.figure('Результат', figsize=(16, 8))
        fig.suptitle('Результаты распознавания', fontsize=16)

        ax1 = fig.add_subplot(1, 4, 1)
        ax2 = fig.add_subplot(2, 4, 2)
        ax3 = fig.add_subplot(2, 4, 3)
        ax4 = fig.add_subplot(2, 4, 4)
        ax5 = fig.add_subplot(2, 4, 6)
        ax6 = fig.add_subplot(2, 4, 7)

        ax7 = plt.axes([0.1, 0.15, 0.2, 0.06])
        button = btn(ax7, 'Остановить', color='gray')
        button.on_clicked(stop)

        plt.text(2.445, 11.555, 'Эталоны', fontsize=14)
                
        for i in range(self.m):
            for j in range(self.t):
                ax1.cla()
                ax1.imshow(Image(i, results[0][i][j][0],
                                 'image').get_image())

                ax1.set_xticks([])
                ax1.set_yticks([])
                ax1.set_title('Тестовое изображение', fontsize=14)
                ax1.set_xlabel('Класс = ' + str(i + 1))

                ax2.cla()
                ax2.imshow(Image(results[0][i][j][1], results[0][i][j][2],
                                 'image').get_image())

                ax2.set_xticks([])
                ax2.set_yticks([])
                ax2.set_title('Гистаграмма яркости')
                ax2.set_xlabel('Класс = ' + str(results[0][i][j][1] + 1))

                ax3.cla()
                ax3.imshow(Image(results[1][i][j][1], results[1][i][j][2],
                                 'image').get_image())

                ax3.set_xticks([])
                ax3.set_yticks([])
                ax3.set_title('DFT')
                ax3.set_xlabel('Класс = ' + str(results[1][i][j][1] + 1))

                ax4.cla()
                ax4.imshow(Image(results[2][i][j][1], results[2][i][j][2],
                                 'image').get_image())

                ax4.set_xticks([])
                ax4.set_yticks([])
                ax4.set_title('DCT')
                ax4.set_xlabel('Класс = ' + str(results[2][i][j][1] + 1))

                ax5.cla()
                ax5.imshow(Image(results[3][i][j][1], results[3][i][j][2],
                                 'image').get_image())

                ax5.set_xticks([])
                ax5.set_yticks([])
                ax5.set_title('Градиент')
                ax5.set_xlabel('Класс = ' + str(results[3][i][j][1] + 1))

                ax6.cla()
                ax6.imshow(Image(results[4][i][j][1], results[4][i][j][2],
                                 'image').get_image())

                ax6.set_xticks([])
                ax6.set_yticks([])
                ax6.set_title('Scale')
                ax6.set_xlabel('Класс = ' + str(results[4][i][j][1] + 1))

                plt.subplots_adjust(wspace=0.3, hspace=0.5, top=0.8)
                plt.show()
                plt.pause(1)
                
                if not self.flag:                                      
                    break
            
            if not self.flag:                                      
                break   

    def show_histograms(self):
        results = self.get_results()
        accuracy = [[] for x in range(5)]

        for i in range(5):
            trues = 0
            alls = 0

            for j in range(len(results[0])):
                for k in range(len(results[0][0])):
                    alls += 1

                    if results[i][j][k][1] == j:
                        trues += 1

                    accuracy[i].append((trues / alls) * 100)

        fig = plt.figure('Графики', figsize=(16, 8))
        fig.suptitle('Точность распознавания методов', fontsize=16)

        ax1 = fig.add_subplot(2, 3, 1)
        ax2 = fig.add_subplot(2, 3, 2)
        ax3 = fig.add_subplot(2, 3, 3)
        ax4 = fig.add_subplot(2, 3, 4)
        ax5 = fig.add_subplot(2, 3, 5)

        y = [[] for x in range(5)]
        x = []

        for i in range(len(accuracy[0])):
            y[0].append(accuracy[0][i])
            y[1].append(accuracy[1][i])
            y[2].append(accuracy[2][i])
            y[3].append(accuracy[3][i])
            y[4].append(accuracy[4][i])
            x.append(i + 1)

            ax1.cla()
            ax1.plot(x, y[0])
            ax1.set_title('Гистаграмма яркости')
            ax1.set_xlabel('Кол-во человек')
            ax1.set_ylabel('Точность (%)')
            ax1.set_yticks(np.arange(0, 110, 10))

            ax2.cla()
            ax2.plot(x, y[1])
            ax2.set_title('DFT')
            ax2.set_xlabel('Кол-во человек')
            ax2.set_ylabel('Точность (%)')
            ax2.set_yticks(np.arange(0, 110, 10))

            ax3.cla()
            ax3.plot(x, y[2])
            ax3.set_title('DCT')
            ax3.set_xlabel('Кол-во человек')
            ax3.set_ylabel('Точность (%)')
            ax3.set_yticks(np.arange(0, 110, 10))

            ax4.cla()
            ax4.plot(x, y[3])
            ax4.set_title('Градиент')
            ax4.set_xlabel('Кол-во человек')
            ax4.set_ylabel('Точность (%)')
            ax4.set_yticks(np.arange(0, 110, 10))

            ax5.cla()
            ax5.plot(x, y[4])
            ax5.set_title('Scale')
            ax5.set_xlabel('Кол-во человек')
            ax5.set_ylabel('Точность (%)')
            ax5.set_yticks(np.arange(0, 110, 10))

            plt.subplots_adjust(wspace=0.3, hspace=0.5)
            plt.show()
            plt.pause(0.1)

# вызов интерфейса
app = QtWidgets.QApplication(sys.argv)
Interface = QtWidgets.QMainWindow()
ui = Ui_Interface()
ui.setupUi(Interface)
Interface.show()
sys.exit(app.exec_())
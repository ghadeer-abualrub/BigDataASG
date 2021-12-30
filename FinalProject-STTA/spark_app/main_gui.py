import io
import folium
from matplotlib.figure import Figure
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from PyQt5 import QtGui, QtWidgets
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtWebEngineWidgets import QWebEngineView
from PyQt5.QtCore import *
from folium import plugins
import pandas as pd


class PlotCanvas(FigureCanvas):
    def __init__(self, parent, topicsCount, topicsCountFull):

        self.fig = Figure(figsize=(7, 3))
        self.ax1 = self.fig.add_subplot(121)
        self.ax2 = self.fig.add_subplot(122)
        FigureCanvas.__init__(self, self.fig)
        self.topicsCount = topicsCount
        self.topicsCountFull = topicsCountFull
        self.setParent(parent)
        # FigureCanvas.setSizePolicy(self, QSizePolicy.Expanding, QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)
        point = parent.rect().bottomRight()
        global_point = parent.mapToGlobal(point)
        self.move(QPoint(635, 16))
        self.setFixedWidth = 612
        self.setFixedHeight = 354
        self.plot()

    def updatePos(self):
        FigureCanvas.updateGeometry(self)

    def plot(self):
        topics = ['Topic 1', 'Topic 2', 'Topic 3']

        self.ax1.bar(topics, self.topicsCount, width=0.3)
        self.ax1.set_title('NLP Results Last month')
        self.ax1.set_ylabel('# of Tweeets')
        self.ax2.bar(topics, self.topicsCountFull, width=0.3)
        self.ax2.set_title('NLP Results Full Period')


class MainWindow(QMainWindow):
    def __init__(self, markers, topicslist, varcount1, mostRetweeted_txt,
                 topicslistFull, varcount2, mostRetweeted_txtF, topicsCount, topicsCountFull, most_fav_txt, most_fav_txtF):
        super(MainWindow, self).__init__()
        self.markers = markers
        self.topicslist = topicslist
        self.varcount1 = varcount1
        self.mostRetweeted_txt = mostRetweeted_txt,
        self.topicslistFull = topicslistFull
        self.varcount2 = varcount2
        self.mostRetweeted_txtF = mostRetweeted_txtF
        self.topicsCount = topicsCount
        self.topicsCountFull = topicsCountFull
        self.most_fav_txt = most_fav_txt
        self.most_fav_txtF = most_fav_txtF

        self.setStyleSheet("background-color: white;")
        self.setGeometry(50, 50, 1310, 700)
        self.setWindowTitle("Spatio-Temporal Tweets Analysis")
        self.initUI()

    def initUI(self):

        self.layout = QVBoxLayout(self)
        self.setLayout(self.layout)

        self.addMap()
        m = PlotCanvas(self, self.topicsCount, self.topicsCountFull)
        m.move = (796, 16)
        m.updatePos()
        self.addSec1()
        self.addSec2()
        # chart = Canvas(self)

    def getCoordinates(self):
        coordinates = self.bounding_box.get_bounds()
        self.parentClass.getTweets()

    def addMap(self):
        self.m = folium.Map(location=[38.9252, -103.8867], tiles='Stamen Terrain',
                            zoom_start=5)
        folium.LatLngPopup().add_to(self.m)

        self.bounding_box = folium.Rectangle(bounds=[(41.9677, -110.6982), (35.6751, -95.9766)],
                                             fill=False,
                                             color='orange',
                                             tooltip='Bounding Box'
                                             ).add_to(self.m)
        color = ''
        for i in range(len(self.markers)):
            if i <= 3:
                color = 'green'
            else:
                color = 'orange'
            folium.Marker(
                location=[self.markers.iloc[i]['lat'],
                          self.markers.iloc[i]['lon']],
                popup=self.markers.iloc[i]['name'],
                icon=folium.Icon(color=color)
            ).add_to(self.m)

        # plot heatmap
        #self.m.add_children(plugins.HeatMap(stationArr, radius=15))
        # save mapdata to data object
        data = io.BytesIO()
        self.m.save(data, close_file=False)
        webView = QWebEngineView(self)
        webView.setFixedWidth(620)
        webView.setFixedHeight(375)
        webView.move(5, 5)
        webView.setHtml(data.getvalue().decode())
        # self.layout.addWidget(webView)
    # ==================================================================================

    def addSec1(self):
        self.label_1 = QLabel('Results between 1-12-2013 and 31-12-2013', self)
        self.label_1.setText(
            '<font color="#404041">Results between </font><font color="red"> 1-12-2013 </font><font color="#404041"> and </font><font color="red"> 31-12-2013 </font>')
        self.label_1.move(16, 390)
        self.label_1.setFixedWidth(620)
        self.label_1.setFixedHeight(36)
        self.label_1.setFont(QFont('Arial', 20))
        self.label_1.setStyleSheet("background-color: #e6e7e8")
        self.bg1 = QLabel(' ', self)
        self.bg1.setStyleSheet("background-color: #e6e7e8")
        self.bg1.setFixedWidth(620)
        self.bg1.setFixedHeight(246)
        self.bg1.move(16, 438)

        topic1 = " "
        topic1 = topic1.join(self.topicslist[0])
        topic2 = " "
        topic2 = topic2.join(self.topicslist[1])
        topic3 = " "
        topic3 = topic3.join(self.topicslist[2])
        self.topics_lbl = QLabel(
            'Topics: \nTopic1: '+topic1+'\nTopic2: '+topic2+'\nTopic3: '+topic3, self)
        self.topics_lbl.move(16, 420)
        self.topics_lbl.setFixedWidth(620)
        self.topics_lbl.setFixedHeight(110)
        self.topics_lbl.setWordWrap(True)
        self.topics_lbl.setFont(QFont('Arial', 12))
        self.topics_lbl.setStyleSheet("background-color: #e6e7e8")

        self.var_lbl = QLabel(
            ' Number of variefied users tweeted about the topics <font color="red">'+str(self.varcount1)+'</font>', self)
        self.var_lbl.move(16, 530)
        self.var_lbl.setFixedWidth(620)
        self.var_lbl.setFixedHeight(36)
        self.var_lbl.setFont(QFont('Arial', 12))
        self.var_lbl.setStyleSheet("background-color: #e6e7e8")
        '''Most Retweeted tweet'''

        self.mostRet = QLabel(
            '<font color="red"> Most Retweeted Tweet:</font> '+str(self.mostRetweeted_txt), self)
        self.mostRet.setWordWrap(True)
        self.mostRet.move(16, 566)
        self.mostRet.setFixedWidth(620)
        self.mostRet.setFixedHeight(80)
        self.mostRet.setFont(QFont('Arial', 12))
        self.mostRet.setStyleSheet("background-color: #e6e7e8")
        '''Most Fav tweet'''
        self.mostFav = QLabel(
            '<font color="red"> Most Favoured Tweet:</font> '+str(self.most_fav_txt), self)
        self.mostFav.setWordWrap(True)
        self.mostFav.move(16, 646)
        self.mostFav.setFixedWidth(620)
        self.mostFav.setFixedHeight(60)
        self.mostFav.setFont(QFont('Arial', 12))
        self.mostFav.setStyleSheet("background-color: #e6e7e8")

    # =======================================================================
    def addSec2(self):
        self.label_2 = QLabel('Results between 12-9-2013 and 31-12-2013', self)
        self.label_2.setText(
            '<font color="#404041">Results between </font><font color="red"> 12-9-2013 </font><font color="#404041"> and </font><font color="red"> 31-12-2013 </font>')
        self.label_2.move(660, 390)
        self.label_2.setFixedWidth(620)
        self.label_2.setFixedHeight(36)
        self.label_2.setFont(QFont('Arial', 20))
        self.label_2.setStyleSheet("background-color: #e6e7e8")

        self.bg2 = QLabel(' ', self)
        self.bg2.setStyleSheet("background-color: #e6e7e8")
        self.bg2.setFixedWidth(620)
        self.bg2.setFixedHeight(246)
        self.bg2.move(660, 438)
        topic1 = " "
        topic1 = topic1.join(self.topicslistFull[0])
        topic2 = " "
        topic2 = topic2.join(self.topicslistFull[1])
        topic3 = " "
        topic3 = topic3.join(self.topicslistFull[2])
        self.topic_lbl2 = QLabel(
            'Topics: \nTopic1: ' + topic1+'\nTopic2: '+topic2+'\nTopic3: '+topic3, self)
        self.topic_lbl2.move(660, 420)
        self.topic_lbl2.setFixedWidth(620)
        self.topic_lbl2.setFixedHeight(110)
        self.topic_lbl2.setWordWrap(True)
        self.topic_lbl2.setFont(QFont('Arial', 12))
        self.topic_lbl2.setStyleSheet("background-color: #e6e7e8")

        self.var_lbl2 = QLabel(
            ' Number of variefied users tweeted about the topics <font color="red">'+str(self.varcount2)+'</font>', self)
        self.var_lbl2.move(660, 530)
        self.var_lbl2.setFixedWidth(615)
        self.var_lbl2.setFixedHeight(36)
        self.var_lbl2.setFont(QFont('Arial', 12))
        self.var_lbl2.setStyleSheet("background-color: #e6e7e8")

        '''Most Retweeted tweet'''
        self.mostRet2 = QLabel(
            '<font color="red"> Most Retweeted Tweet:</font> '+str(self.mostRetweeted_txtF), self)
        self.mostRet2.setWordWrap(True)
        self.mostRet2.move(660, 566)
        self.mostRet2.setFixedWidth(620)
        self.mostRet2.setFixedHeight(88)
        self.mostRet2.setFont(QFont('Arial', 12))
        self.mostRet2.setStyleSheet("background-color: #e6e7e8")
        '''Most Fav tweet'''
        self.mostFav2 = QLabel(
            '<font color="red"> Most Favoured Tweet:</font> '+str(self.most_fav_txtF), self)
        self.mostFav2.setWordWrap(True)
        self.mostFav2.move(660, 646)
        self.mostFav2.setFixedWidth(612)
        self.mostFav2.setFixedHeight(60)
        self.mostFav2.setFont(QFont('Arial', 12))
        self.mostFav2.setStyleSheet("background-color: #e6e7e8")

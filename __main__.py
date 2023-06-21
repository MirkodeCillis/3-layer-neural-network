# GUI implementation of the handwritten number neural network
# there's a canvas where the user can write a number and 
# query the neural network to recognize it.
# The written number is temporarily stored as an image and
# then imported as input of the neural network. 

import numpy as np
from guessNumberNetwork import ngn  # handwritten number neural network
import os                           # delete the temporarily saved image after its use  
import imageio                      # Image management
import random                       # spray effect in canvas
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PIL import Image

# Canvas class description 
class Drawer(QWidget):
    def __init__(self, parent=None):
        QWidget.__init__(self, parent)
        self.setAttribute(Qt.WA_StaticContents)
        h, w = 280, 280
        self.myPenWidth = 5
        self.myPenColor = Qt.black
        self.image = QImage(w, h, QImage.Format_RGB32)
        self.path = QPainterPath()
        self.clearImage()

    def clearImage(self):
        self.path = QPainterPath()
        self.image.fill(Qt.white)  ## switch it to else
        self.update()

    def saveImage(self, fileName, fileFormat):
        self.image.save(fileName, fileFormat)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.drawImage(event.rect(), self.image, self.rect())

    def mouseMoveEvent(self, event):
        SPRAY_PARTICLES = 150
        SPRAY_DIAMETER = 5
        self.path.lineTo(event.pos())
        p = QPainter(self.image)
        p.setPen(QPen(self.myPenColor,
                      self.myPenWidth, Qt.SolidLine, Qt.RoundCap,
                      Qt.RoundJoin))
        # p.drawPath(self.path)
        for n in range(SPRAY_PARTICLES):
            xo = random.gauss(0, SPRAY_DIAMETER)
            yo = random.gauss(0, SPRAY_DIAMETER)
            p.drawPoint(event.x() + xo, event.y() + yo)
        p.end()
        self.update()

    def sizeHint(self):
        return QSize(280, 280)

def clear():
    drawer.clearImage()
    sect2.hide()
    pnl_correction.hide()
    risposta_giusta.clear()
    w.adjustSize()

# Handwritter number query: btnExec button
def execNN():
    global img_data, value
    # save the image...
    drawer.saveImage("input.png", "PNG")
    # ... and resize it for the matrix dimension (28x28)
    img = Image.open("input.png")
    hsize, basewidth = 28, 28
    img = img.resize((basewidth, hsize), Image.ANTIALIAS)
    img.save("input.png")
    # resize of input values: png pixels values are
    # from 0 (black) to 255 (white)
    img_array = imageio.imread("input.png", as_gray=True)
    img_data = 255.0 - img_array.reshape(784)
    img_data = (img_data / 255.0 * 0.99) + 0.01
    output_list = ngn.query(img_data)
    value = np.argmax(output_list)
    risultato.setText("Risposta della rete neurale: " + str(value))
    sect2.show()
    w.adjustSize()
    os.remove("input.png")


# right answer: right_btn buttone
def right():
    global totale, corretti
    target_list = np.zeros(ngn.onodes) + 0.01
    target_list[value] = 0.99
    ngn.train(img_data, target_list)
    corretti = corretti + 1
    totale = totale + 1
    clear()


# wrong answer: wrong_btn button
def wrong():
    global totale
    totale = totale + 1
    pnl_correction.show()
    w.adjustSize()
    pass

# improves the answer with the one given: bottone send
def repeat():
    global value, corretti
    value = int(risposta_giusta.text())
    if value < 0 or value >= ngn.onodes:
        QErrorMessage(risp_corr_box).showMessage("Inseririre un numero compreso fra 0 e " + str(ngn.onodes-1))
        return
    right()
    corretti = corretti - 1
    pass

corretti, totale, value, img_data = 0, 0, -1, np.zeros(ngn.inodes)
# GUI setup
app = QApplication([])
w = QWidget()

# section 1: drawer + buttons
drawer = Drawer()
sect1 = QGroupBox()
sect1.setLayout(QVBoxLayout())
sect1.setStyleSheet("border: 0")
# layout dei bottoni "verifica" e "reset"
pnl_ex_cl = QGroupBox()
pnl_ex_cl.setStyleSheet("border: 0")
pnl_ex_cl.setLayout(QHBoxLayout())
btnExec = QPushButton("Verifica il numero inserito")
btnClear = QPushButton("Reset")
btnExec.setStyleSheet("background-color: #141651; color: white; padding: 5")
btnClear.setStyleSheet("background-color: #141651; color: white; padding: 5")
btnExec.clicked.connect(lambda: execNN())
btnClear.clicked.connect(clear)
pnl_ex_cl.layout().addWidget(btnExec)
pnl_ex_cl.layout().addWidget(btnClear)
sect1.layout().addWidget(drawer)
sect1.layout().addWidget(pnl_ex_cl)

# section 2: answer and validation
sect2 = QGroupBox()
sect2.setLayout(QVBoxLayout())
sect2.setStyleSheet("border: 0")
risultato = QLabel()
# panel for validation of the answer (right or wrong)
pnl_answer = QGroupBox()
pnl_answer.setLayout(QHBoxLayout())
right_btn = QPushButton("Sì")
right_btn.setStyleSheet("background-color: green; color: white; padding: 5")
right_btn.clicked.connect(right)
wrong_btn = QPushButton("No")
wrong_btn.setStyleSheet("background-color: red; color: white; padding: 5")
wrong_btn.clicked.connect(wrong)
pnl_answer.layout().addWidget(right_btn)
pnl_answer.layout().addWidget(wrong_btn)
pnl_answer.setStyleSheet("border: 0")
# correction panel
pnl_correction = QGroupBox()
pnl_correction.setLayout(QVBoxLayout())
pnl_correction.setStyleSheet("border: 0")
risp_corr_box = QGroupBox()
risp_corr_box.setLayout(QHBoxLayout())
risp_corr_box.setStyleSheet("border: 0")
risposta_giusta = QLineEdit()
send = QPushButton("Improve and repeat")
send.setStyleSheet("background-color: #141651; color: white; padding: 5")
send.clicked.connect(repeat)
risp_corr_box.layout().addWidget(QLabel("Right answer: "))
risp_corr_box.layout().addWidget(risposta_giusta)
pnl_correction.layout().addWidget(risp_corr_box)
pnl_correction.layout().addWidget(send)

sect2.layout().addWidget(risultato)
sect2.layout().addWidget(QLabel("Is the answer correct?"))
sect2.layout().addWidget(pnl_answer)
sect2.layout().addWidget(pnl_correction)



if __name__ == '__main__':
    # final setup and execution
    w.setLayout(QVBoxLayout())
    w.setStyleSheet("border: 0; font-size: 15px")
    w.layout().addWidget(sect1)
    w.layout().addWidget(sect2)
    sect2.hide()
    pnl_correction.hide()

    w.show()
    app.exec_()
    print((float(corretti / totale) * 100), "% corretti.", sep="")
    ngn.exportWeights("weights.txt")

import sys
from PyQt5.QtWidgets import QApplication, QGraphicsWidget, QMainWindow, QFileDialog
import PyQt5
from pyqtgraph import PlotWidget, plot
import pyqtgraph
from form import Ui_MainWindow
import graph
from TimeDomainHRV import TimeDomainHRV
import pyqtgraph as pg
import defaults

# This value has to take into account the sliders min/max values
SLIDE_SCALE_FACTOR = 100

class popup_graph_window(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Plotter")
        self.central_widget = PlotWidget(self)
        self.setGeometry(100, 100, 800, 600)
        self.setCentralWidget(self.central_widget)

class sub_window(QMainWindow):
    def __init__(self):
        super(sub_window, self).__init__()
        self.setWindowTitle("sub window")
        self.ui = graph.Ui_MainWindow()
        self.ui.setupUi(self) 
        # self.setGeometry(100, 100, 800, 600)


class HRV_GUI(QMainWindow): # Main GUI window
    def __init__(self):
        super(HRV_GUI, self).__init__()

        # Set up ui from designer
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)

        self.initUI()

        # Create our HRV instance
        self.myHRV = TimeDomainHRV()


    def initUI(self):
        # TODO: look into creating lambda functions to reduce methods created
        self.ui.button1.clicked.connect(self.button1_clicked)
        self.ui.button2.clicked.connect(self.button2_clicked)
        self.ui.button3.clicked.connect(self.button3_clicked)
        self.ui.button4.clicked.connect(self.button4_clicked)
        self.ui.button5.clicked.connect(self.button5_clicked)
        self.ui.button6.clicked.connect(self.button6_clicked)
        
        self.ui.checkBox.stateChanged.connect(self.checkBox_changed)
        # self.region = pyqtgraph.LinearRegionItem(brush=(255, 251, 100, 140) , hoverBrush=(225, 40, 40, 140), pen={'color': (255, 89, 89, 200), 'width': 5}, hoverPen='r')
        # self.ui.graph_widget.addItem(self.region)
        # self.region.sigRegionChanged.connect(self.region_selection_change)
        self.ui.file_location_box.setPlainText(defaults.ecg_file_path) # todo: const file

        # Setup for tab2 
        self.set_params_to_default()
        self.ui.slider1.valueChanged.connect(self.slider1_moved)
        self.ui.slider1_box.valueChanged.connect(self.slider1_box_changed)
        self.ui.slider2.valueChanged.connect(self.slider2_moved)
        self.ui.slider2_box.valueChanged.connect(self.slider2_box_changed)
        self.ui.slider3.valueChanged.connect(self.slider3_moved)
        self.ui.slider3_box.valueChanged.connect(self.slider3_box_changed)
        self.ui.slider4.valueChanged.connect(self.slider4_moved)
        self.ui.slider4_box.valueChanged.connect(self.slider4_box_changed)
        self.ui.slider5.valueChanged.connect(self.slider5_moved)
        self.ui.slider5_box.valueChanged.connect(self.slider5_box_changed)
        self.ui.slider6.valueChanged.connect(self.slider6_moved)
        self.ui.slider6_box.valueChanged.connect(self.slider6_box_changed)


    def button1_clicked(self):
        self.myHRV.calculate_peaks()
        self.myHRV.compute()
        self.ui.output_box_1.append(self.myHRV.print_s())
        self.ui.graph_widget.plot(self.myHRV.time, self.myHRV.ECG_data)

        if self.ui.checkBox_2.isChecked():
            print("start time = " + str(self.myHRV.time[0]))
            print("end time = " + str(self.myHRV.time[-1]))
            self.raw_ECG_plot = popup_graph_window()
            self.raw_ECG_plot.central_widget.plot(self.myHRV.time, self.myHRV.ECG_data)
            self.raw_ECG_plot.show() 
        if self.ui.checkBox_3.isChecked():
            self.RRI_plot = popup_graph_window()
            self.RRI_plot.central_widget.plot(self.myHRV.peaks, self.myHRV.ECG_data[self.myHRV.peaks], symbol = 'x')
            self.ui.graph_widget.plot(self.myHRV.peaks, self.myHRV.ECG_data[self.myHRV.peaks], symbol = 'x')
            self.RRI_plot.show() 

    def button2_clicked(self):
        r = self.region.getRegion()
        # TODO: edit the function so it dosent read from file
        if (r[0] < self.myHRV.file.channels[1].time_index[0] or r[1] > self.myHRV.file.channels[1].time_index[-1]):
            self.ui.output_box_1.append("Region selected is out of bounds of the signal. Please select a valid region of time.")
        else:
            self.myHRV.set_region(r)
            self.ui.output_box_1.append("Region set from: " + str(r[0]) + " to: " + str(r[1]))

    def button3_clicked(self):
        # TODO: make sure it only opens the correct files or promts the user at least
        fname = QFileDialog.getOpenFileName(self, 'Open ECG file')
        self.ui.file_location_box.setPlainText(fname[0])
        self.myHRV.set_input_file(fname[0])

    def button4_clicked(self):
        pulled_text = self.ui.file_location_box.toPlainText()
        self.myHRV.set_input_file(pulled_text)
    
    def button5_clicked(self):
        # In order to keep the window alive, it has to be a member of the class, or something similar
        # Since if its only in the scope of the function
        # the desctructor will be called and the new window will die
        # otherwise, that would be a memory leak
        # self.w = popup_graph_window()
        # self.w.show() 

        self.new_window = sub_window()
        self.new_window.ui.centralwidget.plot(self.myHRV.time, self.myHRV.ECG_data)
        self.new_window.show()
        # self.meow = graph.Ui_MainWindow()

    def button6_clicked(self):
        self.set_params_to_default()

    def checkBox_changed(self):
        if self.ui.checkBox.isChecked():
            self.region = pyqtgraph.LinearRegionItem(brush=(255, 251, 100, 140), 
                hoverBrush=(225, 40, 40, 140), pen={'color': (255, 89, 89, 200), 
                'width': 5}, hoverPen='r', 
                bounds=[self.myHRV.file.channels[1].time_index[0], 
                self.myHRV.file.channels[1].time_index[-1]])
            self.ui.graph_widget.addItem(self.region)
            self.region.sigRegionChanged.connect(self.region_selection_change)
        else:
            self.region.sigRegionChanged.disconnect()
            self.ui.graph_widget.removeItem(self.region)

    def slider1_moved(self):
        self.ui.slider1_box.setValue(self.ui.slider1.value())
        self.myHRV.set_BP_height(self.ui.slider1.value())

    def slider1_box_changed(self):
        self.ui.slider1.setValue(self.ui.slider1_box.value())
        self.myHRV.set_BP_height(self.ui.slider1_box.value())

    def slider2_moved(self):
        self.ui.slider2_box.setValue(self.ui.slider2.value())
        self.myHRV.set_BP_distance(self.ui.slider2.value())

    def slider2_box_changed(self):
        self.ui.slider2.setValue(self.ui.slider2_box.value())
        self.myHRV.set_BP_distance(self.ui.slider2_box.value())

    def slider3_moved(self):
        self.ui.slider3_box.setValue(self.ui.slider3.value() / SLIDE_SCALE_FACTOR)
        self.myHRV.set_ECG_height(self.ui.slider3.value() / SLIDE_SCALE_FACTOR)

    def slider3_box_changed(self):
        self.ui.slider3.setValue(int(self.ui.slider3_box.value() * SLIDE_SCALE_FACTOR))
        self.myHRV.set_ECG_height(self.ui.slider3_box.value())

    def slider4_moved(self):
        self.ui.slider4_box.setValue(self.ui.slider4.value())
        self.myHRV.set_ECG_distance(self.ui.slider4.value())

    def slider4_box_changed(self):
        self.ui.slider4.setValue(self.ui.slider4_box.value())
        self.myHRV.set_ECG_distance(self.ui.slider4_box.value())

    def slider5_moved(self):
        self.ui.slider5_box.setValue(self.ui.slider5.value())
        self.myHRV.set_BP_prominence(self.ui.slider5.value())

    def slider5_box_changed(self):
        self.ui.slider5.setValue(self.ui.slider5_box.value())
        self.myHRV.set_BP_prominence(self.ui.slider5_box.value())

    def slider6_moved(self):
        self.ui.slider6_box.setValue(self.ui.slider6.value() / SLIDE_SCALE_FACTOR)
        self.myHRV.set_ECG_prominence(self.ui.slider6.value() / SLIDE_SCALE_FACTOR)

    def slider6_box_changed(self):
        self.ui.slider6.setValue(int(self.ui.slider6_box.value() * SLIDE_SCALE_FACTOR))
        self.myHRV.set_ECG_prominence(self.ui.slider6_box.value())

    def set_params_to_default(self):
        self.ui.slider1.setValue(defaults.BP_height)
        self.ui.slider1_box.setValue(defaults.BP_height)
        self.ui.slider2.setValue(defaults.BP_distance)
        self.ui.slider2_box.setValue(defaults.BP_distance)
        self.ui.slider5.setValue(defaults.BP_prominence)
        self.ui.slider5_box.setValue(defaults.BP_prominence)
        # Sliders dont take floats? :(
        self.ui.slider3.setValue(int(defaults.ECG_height * SLIDE_SCALE_FACTOR))  
        self.ui.slider3_box.setValue(defaults.ECG_height)
        self.ui.slider4.setValue(defaults.ECG_distance)
        self.ui.slider4_box.setValue(defaults.ECG_distance)
        self.ui.slider6.setValue(int(defaults.ECG_prominence * SLIDE_SCALE_FACTOR))
        self.ui.slider6_box.setValue(defaults.ECG_prominence)

    def region_selection_change(self):
        r = self.region.getRegion()
        self.ui.label_10.setText(str(r[0]))
        self.ui.label_11.setText(str(r[1]))
    
app = QApplication(sys.argv)
window = HRV_GUI()
window.show()
sys.exit(app.exec_())

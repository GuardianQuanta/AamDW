
import numpy as np
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *

def TableFloatItemSetHelper(Value,Editable=False,RoundNearest=4,isNan =False,_background_color=None):
    NewItem = QTableWidgetItem()
    if not isNan:
        NewItem.setData(Qt.EditRole, float(np.round(Value,RoundNearest)))
        # NewItem.setData(Qt.EditRole, QVariant(Value))
    if Editable:
        NewItem.setFlags(Qt.ItemIsSelectable | Qt.ItemIsEnabled | Qt.ItemIsEditable)
    else:
        NewItem.setFlags(Qt.ItemIsSelectable | Qt.ItemIsEnabled)

    if _background_color is not None:
        NewItem.setBackground(_background_color)
    return NewItem


class FloatDelegate(QItemDelegate):
    def __init__(self, decimals, parent=None):
        QItemDelegate.__init__(self, parent=parent)
        self.nDecimals = decimals

    def paint(self, painter, option, index):
        value = index.model().data(index, Qt.EditRole)
        try:
            number = float(value)
            painter.drawText(option.rect, Qt.AlignLeft, "{:.{}f}".format(number, self.nDecimals))
        except :
            QItemDelegate.paint(self, painter, option, index)

class mySpinBoxDelegate(QItemDelegate):
    def __init__(self, decimals, parent=None):
        QItemDelegate.__init__(self, parent=parent)
        self.nDecimals = decimals

    def createEditor(self, parent: QWidget, option: 'QStyleOptionViewItem', index) -> QWidget:
        editor = QDoubleSpinBox(parent)
        editor.setDecimals(self.nDecimals)
        editor.setRange(-999999999,999999999)

        return editor


def try_parse_float_tableItem(item):
    if item is None:
        return False,None
    if item.text() == '':
        return False,None
    try:
        a= float(item.text())
        return True,a
    except ValueError:
        return False,None

class myQButton(QWidget):
    def __init__(self, _qty_value,_function ,parent=None, ):
        QWidget.__init__(self, parent)
        self.button = QPushButton(self)
        self.button.setText(str(_qty_value))
        # self.button
        self.qty_value = _qty_value
        self.button.clicked.connect(self.on_button_click)
        self.function = _function

        self.setSizePolicy(QSizePolicy.MinimumExpanding,QSizePolicy.Preferred)

    def on_button_click(self):
        self.function(self.qty_value)


class ButtonBlock(QWidget):

    def __init__(self,main_function, *args):
        super(QWidget, self).__init__()
        grid = QGridLayout(self)

        list = np.array([1,2,3,4,5,6,7,8,9,10,20])
        list_neg = -1*list
        self.main_function = main_function
        ButtonNum_array = np.hstack( (list.reshape(-1,1),list_neg.reshape(-1,1)) )

        self.InitReady = False

        for col_i in range(ButtonNum_array.shape[1]):
            for row_i in range(ButtonNum_array.shape[0]):
                button = QPushButton(str(ButtonNum_array[row_i,col_i]), self)
                # button = myQButton(str(ButtonNum_array[row_i,col_i]),self.make_calluser, self)
                button.clicked.connect(self.make_calluser(ButtonNum_array[row_i,col_i]))
                # button.clicked.connect( lambda: self.make_calluser(ButtonNum_array[row_i,col_i]) )
                # row, col = divmod(i, 5)
                grid.addWidget(button, row_i, col_i)
        # self.setLayout(grid)
    def make_calluser(self, name):
        def calluser():
            if self.InitReady:
                print(name)
                self.main_function(name)
        # return 1
        return calluser # .connect expects a function return otherwise it doesn't like it


class PermanentMenu(QMenu):
    def hideEvent(self, event):
        self.show()



class Instrument_Selection_Menu(QWidget):

    def __init__(self,UnderGroup,_addtrade_table,_perm_menu, *args):
        super(QWidget, self).__init__()
        # instrument_selection_menu = PermanentMenu(self)
        self.UnderGroup = UnderGroup
        self.addtrade_table = _addtrade_table
        self.instrument_selection_menu = _perm_menu

        self.InitReady = False

        instr_menu = self.instrument_selection_menu.addMenu("Option Selection")
        for k,v in UnderGroup.Maturity_Dict_index.items():
            month_menu = instr_menu.addMenu(v.MaturityDate.strftime('%b%y'))

            SeriesCode = v.MonthChar + v.MaturityDate.strftime('%y')

            put_menu = month_menu.addMenu("P")
            call_menu = month_menu.addMenu("C")
            for strike, optionpair in v.OptionPairMap.items():
                # print()
                put_action = put_menu.addAction(str(strike))
                put_action.triggered.connect(self.instrument_menu_callback_factory("S50_"+ SeriesCode+"_P"+str(strike)))
                call_action =call_menu.addAction(str(strike))
                call_action.triggered.connect(self.instrument_menu_callback_factory("S50_"+ SeriesCode+"_C"+str(strike)))

        self.InitReady = True
        # self.show()
        # self.setLayout(grid)
    def instrument_menu_callback_factory(self, name):
        # def make_call():
        #     if self.InitReady:
        #         print(name)
        #         self.addtrade_table.setItem(0,0,QTableWidgetItem(name))
        # # return 1
        # return make_call # .connect expects a function return otherwise it doesn't like it
        return lambda: self.addtrade_table.setItem(0,0,QTableWidgetItem(name))

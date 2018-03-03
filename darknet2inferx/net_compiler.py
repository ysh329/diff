"""
    net_compiler.py
    Copyright 2017 Junhui Zhang <mrlittlezhu@gmail.com>
    Portions Copyright 2017 Xianyi Zhang <http://xianyi.github.io> and Chaowei Wang <wangchaowei@ncic.ac.cn>

This script made to build caffe net protobuf file to inferxlite[website] .c file

"""

__author__ = "Junhui Zhang <https://mrlittlepig.github.io>"
__version__ = "0.1"
__date__ = "March 4,2017"
__copyright__ = "Copyright: 2017 Junhui Zhang; Portions: 2017 Xianyi Zhang <http://xianyi.github.io>; Portions: 2017 Chaowei Wang;"

import re
import sys
from optparse import OptionParser
from abc import abstractmethod


def cformatparam(string_param):
    cformat = ""
    for cha in string_param:
        if re.match('^[0-9a-zA-Z]+$', cha):
            cformat += cha
    return cformat

def isac(c):
    """
    A simple function, which determine whether the
    string element char c is belong to a decimal number
    :param c: a string element type of char
    :return: a bool type of determination
    """
    try:
        int(c)
        return True
    except:
        if c == '.' or c == '-' or c == 'e':
            return True
        else:
            return False

def hasannotation(string_list):
    """
    Judge whether the string type parameter string_list contains annotation
    """
    for c in string_list:
        if c == "#":
            return True
    return False

def dropannotation(annotation_list):
    """
    Drop out the annotation contained in annotation_list
    """
    target = ""
    for c in annotation_list:
        if not c == "#":
            target += c
        else:
            return target
    return target

class LayerFactory(object):
    """
    Layer factory used to connect layer and sublayer.

    Members
    ----------
    __layer_register: a list to store layer type, which is registered in layer system.
    layer_string: contain a whole layer information.
    type: all the layers type are included in __layer_register.
    layer: which sotres layer object by __gen_layer__ function, using
        statement exec ('self.layer = %s(self.layer_string)'%self.__type)
        as self.layer = Convolution(self.layer_string) an example.
    ----------
    """

    __layer_register = ['Input', 'Convolution', 'Deconvolution', 'Pooling',
                        'Crop', 'Eltwise', 'ArgMax', 'BatchNorm', 'Concat',
                        'Scale', 'Sigmoid', 'Softmax', 'TanH', 'ReLU', 'LRN',
                        'InnerProduct', 'Dropout','Reshape']

    def __init__(self, layer_string=None,net_name=None):
        self.layer_string = layer_string
        self.type = None
       
        self.net_name = net_name
        self.layer = None
        self.__init_type__()
        self.__gen_layer__()

    def __init_type__(self):
        phase_list = self.layer_string.split('type')
        phase_num = len(phase_list)
        if phase_num == 1:
            self.type = "Input"
        elif phase_num >= 2:
            self.type = phase_list[1].split('\"')[1]

    def __gen_layer__(self):
        if self.type in self.__layer_register:
            exec ('self.layer = %s(self.layer_string,self.net_name)'%self.type)
        else:
            TypeError("Type {} layer is not in layer register".format(self.type))

class Layer(object):
    """Layer parent class"""

    __phases_string = ['name', 'type', 'bottom', 'top']
    modelstr="model"
    datastr="data"
    pdata="pdata"
    context="context_id"
    def __init__(self, layer_string=None,net_name=None):
        self.layer_string = layer_string
        self.type = None
        self.name = None
        self.bottom = None
        self.top = None
 
        context="context_id"
        self.net_name = net_name	
        self.bottom_layer = None
        self.num_input = None
        self.num_output = None

        self.interface_c = None
        self.interface_criterion = None
        self.other = None
        self.__init_string_param__()
        self.__init_top__()
        self.__list_all_member__()
    @abstractmethod
    def __calc_ioput__(self):
        """Calculate num_input and num_output"""
        pass

    @abstractmethod
    def __interface_c__(self):
        """Write the predestinate parameter into c type layer function"""
        pass

    def __debug_print__(self, string_list, printout=False):
        """Choose to print or not controled by printout"""
        if printout:
            print(string_list)

    def __init_bottom__(self):
        """Sometimes a layer has more than one bottom, so we pull it out alone"""
        bottoms_tmp = self.layer_string.split('bottom')
        bottom_num = len(bottoms_tmp)
        bottoms = []
        if bottom_num == 1:
            self.bottom = None
        else:
            for index in range(1, bottom_num):
                bottoms.append(bottoms_tmp[index].split('\"')[1]+"_data")
            self.bottom = bottoms
    
    def __init_top__(self):
        self.top += "_data"

    def __init_string_param__(self):
        """
        String parameters like name: "layername", key is name the value
        is the string type "layername", this function finds string parameters,
        which are stored in private list __phases_string, then stores the keys
        values in member variables by using exec function.
        """
        for phase in self.__phases_string:
            if phase == 'bottom':
                self.__init_bottom__()
                continue
            phase_list = self.layer_string.split(phase)
            phase_num = len(phase_list)
            if phase_num == 1:
                continue
            elif phase_num == 2:
                exec ('self.%s=phase_list[1].split(\'\"\')[1]' % phase)
            else:
                member = []
                for index in range(1, phase_num):
                    member.append(phase_list[index].split('\"')[1])
                exec ('self.%s=member' % phase)
                if phase == "type":
                    self.type = self.type[0]
        self.__debug_print__("Init string param.")

    def __init_number_param__(self, phases_number):
        """
        Number parameters like num_output: 21, key is num_output the value
        is the number 21, this function finds number parameters, which are
        stored in list phases_number, then stores the keys values in member
         variables by using exec function.
        """
        for phase in phases_number:
            phase_list = self.layer_string.split(phase)
            phase_num = len(phase_list)
            if phase_num == 1:
                continue
            elif phase_num == 2:
                exec ('self.%s = self.__find_all_num__(phase_list[1])[0]' % phase)
            else:
                print("Error phase_num:%d" % phase_num)
        self.__debug_print__("Init number param.")

    def __init_decimal_param__(self, phases_decimal):
        """
        Decimal parameters like eps: 0.0001, key is eps the value is the
        decimal 0.0001, this function finds decimal parameters, which are
        stored in list phases_decimal, then stores the keys values in member
         variables by using exec function.
        """
        for phase in phases_decimal:
            phase_list = self.layer_string.split(phase)
            phase_num = len(phase_list)
            if phase_num == 1:
                continue
            elif phase_num >= 2:
                exec ('self.%s = []' % phase)
                for index in range(1, phase_num):
                    exec ('self.%s.append(self.__find_first_decimal__(phase_list[index].split(\':\')[1]))' % phase)
        self.__debug_print__("Init decimal param.")

    def __init_binary_param__(self, phase, default='false'):
        """
        Binary parameters like bias_term: false, key is bias_term the value
        is the bool type false, this function finds binary parameter, which
        pass in as phase, then stores the keys values in member variable by
        using exec function. Parameter default to set the default satus of
        the phase parameter
        """
        if default == 'false':
            neg_default = 'true'
        else:
            neg_default = 'false'
        phase_list = self.layer_string.split(phase)
        phase_num = len(phase_list)
        if phase_num == 1:
            exec ('self.%s = \'%s\'' % (phase, default))
        elif phase_num >= 2:
            if len(phase_list[1].split(':')[1].split(default)) == 1:
                exec ('self.%s = \'%s\'' % (phase, neg_default))
            else:
                exec ('self.%s = \'%s\'' % (phase, default))

    def __find_all_num__(self, string_phase):
        """
        A function to find series of numbers
        :param string_phase: string type key like num_output
        :return: a list stores numbers found in string_phase
        """
        number = re.findall(r'(\w*[0-9]+)\w*', string_phase)
        return number

    def __find_first_decimal__(self, string_phase):
        """
        A function to find series of decimal
        :param string_phase: string type key like moving_average_fraction
        :return: a list stores decimals found in string_phase
        """
        decimals = ""
        for index in range(len(string_phase)):
            if isac(string_phase[index]):
                decimals += string_phase[index]
            else:
                decimals += ' '
        for decimal in decimals.split(' '):
            if not decimal == '':
                return decimal


    def __list_all_member__(self, listout=False):
        """Show all member variables"""
        if listout:
            for name, value in vars(self).items():
                if value == None:
                    continue
                self.__debug_print__('%s = %s' % (name, value),printout=True)



class Input(Layer):
    """Input layer"""

    __phases_string = ['name', 'type', 'top']
    def __init__(self, layer_string=None,net_name=None):
        Layer.__init__(self, layer_string,net_name)
        	
        self.dim = []
        self.__init_dim__()
        self.__list_all_member__()

    def __init_dim__(self):
        phase_list = self.layer_string.split('dim:')
        phase_num = len(phase_list)
        if phase_num == 1:
            self.__debug_print__("Input layer %s has no input dims" % self.name, printout=True)
        elif phase_num >= 2:
            for index in range(1, phase_num):
                self.dim.append(self.__find_all_num__(phase_list[index]))

    def __init_string_param__(self):
        if len(self.layer_string.split("type")) == 1 \
            and len(self.layer_string.split("top")) == 1\
                and len(self.layer_string.split("dim:")) >= 2:
            self.name = "data"
            self.type = "Input"
            self.top = "data"
            return
        for phase in self.__phases_string:
            if phase == 'bottom':
                self.__init_bottom__()
                continue
            phase_list = self.layer_string.split(phase)
            phase_num = len(phase_list)
            if phase_num == 1:
                continue
            elif phase_num == 2:
                exec ('self.%s=phase_list[1].split(\'\"\')[1]' % phase)
            else:
                member = []
                for index in range(1, phase_num):
                    member.append(phase_list[index].split('\"')[1])
                exec ('self.%s=member' % phase)
        self.__debug_print__("Init string param.")

    def __interface_c__(self):
        self.interface_criterion = \
            "Input(int dim1,int dim2,int dim3,int " \
            "dim4,char *top,char *name)"
        self.interface_c = "Input("
       # for d in self.dim:
       #     self.interface_c += "{}".format(d[0])
       #     self.interface_c += ','
        self.interface_c += "{},".format("nchw")
        self.interface_c += '{},'.format(Layer.pdata)
        self.interface_c += '\"{}\",'.format(self.top)
        self.interface_c += '\"{}\",'.format(self.name)
        self.interface_c += '{},'.format(Layer.modelstr)
        self.interface_c += '{});'.format(Layer.datastr)

    def __calc_ioput__(self):
        self.num_input = None
        self.__debug_print__(self.name)
        self.num_output = int(self.dim[1][0])


class Convolution(Layer):
    """Convolution layer"""

    __phases_number = ['num_output', 'kernel_size', 'stride', 'pad',
                       'group', 'dilation', 'axis']
    __phases_binary = ['bias_term', 'force_nd_im2col']
    def __init__(self, layer_string=None,net_name=None):
        self.group = 1
        self.axis = 1
        self.kernel_size = None
        self.dilation = 1
        self.stride = 1
        self.pad = 0
        self.bias_term = 'true'
        self.merge_bn = False
        self.force_nd_im2col = 'false'
        Layer.__init__(self, layer_string,net_name)
        self.__init_number_param__(self.__phases_number)
        self.__init_binary_param__(self.__phases_binary[0], default='true')
        self.__init_binary_param__(self.__phases_binary[1], default='false')
        self.__list_all_member__()

        self.kernel_h = self.kernel_size
        self.kernel_w = self.kernel_size
        self.stride_h = self.stride
        self.stride_w = self.stride
        self.pad_h = self.pad
        self.pad_w = self.pad

    def __interface_c__(self):
        if(self.merge_bn):
            self.__set_bias_term__() 
        self.interface_criterion = \
            "Convolution(int num_input,int num_output,int kernel_h,int kernel_w,int stride_h," \
            "int stride_w,int pad_h,int pad_w,int group,int dilation,int axis," \
            "bool bias_term,bool force_nd_im2col,char *bottom,char *top, char *name)"
        #print("bias_term: ")
        #print(self.bias_term)
        self.interface_c = "inferx_convolution("
        self.interface_c += "{},{},{},{},{},{},{},{}".\
            format(self.num_input,self.num_output,self.kernel_h,self.kernel_w,
                   self.stride_h,self.stride_w,self.pad_h,self.pad_w)
        self.interface_c += ",{},{},{}".format(self.group,self.dilation,self.axis)
        self.interface_c += ",{},{}".format(self.bias_term,self.force_nd_im2col)
        #self.interface_c += ",\"{}\",\"{}\",\"{}\",{},{});".format(self.bottom_layer[0].top,self.top,self.name,Layer.modelstr,Layer.datastr)
        self.interface_c += ",\"{}\",\"{}\",\"{}\",{},{},".format(self.bottom_layer[0].top,self.top,self.name,Layer.modelstr,Layer.datastr)

    def __calc_ioput__(self):
        self.num_input = self.bottom_layer[0].num_output
    
    def __set_bias_term__(self):
        self.bias_term ='true'
    def __set_merge_bn__(self):
        self.merge_bn= True
 
class Deconvolution(Convolution):
    """Deconvolution layer"""

    __phases_number = ['num_output', 'kernel_size', 'stride', 'pad', 'dilation']
    def __init__(self, layer_string=None,net_name=None):
        Convolution.__init__(self, layer_string,net_name=None)

    def __interface_c__(self):
        self.interface_criterion = \
            "Deconvolution(int num_input,int num_output,int kernel_h,int kernel_w," \
            "int stride_h,int stride_w,int pad_h,int pad_w,int group,int dilation,int axis," \
            "bool bias_term,bool force_nd_im2col,char *bottom,char *top,char *name)"
        self.interface_c = "inferx_deconvolution("
        self.interface_c += "{},{},{},{},{},{},{},{}". \
            format(self.num_input, self.num_output, self.kernel_h, self.kernel_w,
                   self.stride_h, self.stride_w, self.pad_h, self.pad_w)
        self.interface_c += ",{},{},{}".format(self.group, self.dilation, self.axis)
        self.interface_c += ",{},{}".format(self.bias_term, self.force_nd_im2col)
        self.interface_c += ",\"{}\",\"{}\",\"{}\",{},{});".format(self.bottom_layer[0].top, self.top, self.name,Layer.modelstr,Layer.datastr)


class Pooling(Layer):
    """Pooling layer"""

    __phases_number = ['kernel_size', 'stride', 'pad']
    __phases_binary = ['global_pooling']
    __pool_phases = ['MAX', 'AVE', 'STOCHASTIC']
    def __init__(self, layer_string=None,net_name=None):
        Layer.__init__(self, layer_string,net_name)
        self.kernel_size = None
        self.stride = 1
        self.pool = 'MAX'
        self.global_pooling = 'false'
        self.pad = 0

        self.__init_number_param__(self.__phases_number)
        self.__init_binary_param__(self.__phases_binary[0], default='false')
        self.__init_pool__()
        self.__list_all_member__()

        self.kernel_h = self.kernel_size
        self.kernel_w = self.kernel_size
        self.stride_h = self.stride
        self.stride_w = self.stride
        self.pad_h = self.pad
        self.pad_w = self.pad

    def __init_pool__(self):
        for phase in self.__pool_phases:
            phase_list = self.layer_string.split(phase)
            phase_num = len(phase_list)
            if phase_num == 1:
                self.__debug_print__("Pooling layer %s has no pool method %s." % (self.name, phase))
                continue
            elif phase_num == 2:
                self.pool = phase
                self.__debug_print__("Pooling layer %s has pool method %s." % (self.name, self.pool))
            else:
                self.__debug_print__("Pool layer method error.",printout=True)

    def __calc_ioput__(self):
        self.num_input = self.bottom_layer[0].num_output
        self.num_output = self.num_input

    def __interface_c__(self):
        if self.global_pooling == 'true':
            self.interface_criterion = \
                "inferx_globalPooling(enum PoolMethod pool,char *bottom,char *top,char *name)"
            self.interface_c = "GlobalPooling("
        else:
            self.interface_criterion = \
                "inferx_pooling(int kernel_h,int kernel_w,int stride_h,int stride_w,int pad_h," \
                "int pad_w,enum PoolMethod pool,char *bottom,char *top,char *name)"
            self.interface_c = "Pooling("
            self.interface_c += "{},{},{},{},{},{},". \
                format(self.kernel_h,self.kernel_w,
                       self.stride_h,self.stride_w,self.pad_h,self.pad_w)
        self.interface_c += "{}".format(self.pool)
        self.interface_c += ",\"{}\",\"{}\",\"{}\",{},{});".format(self.bottom_layer[0].top, self.top, self.name,Layer.modelstr,Layer.datastr)

class Crop(Layer):
    """Crop layer"""

    __phases_number =  ['axis', 'offset']
    def __init__(self, layer_string=None,net_name=None):
        Layer.__init__(self, layer_string,net_name)
        self.axis = 2
        self.offset = 0

        self.__init_number_param__(self.__phases_number)
        self.__list_all_member__()

    def __calc_ioput__(self):
        self.num_input = self.bottom_layer[0].num_output
        self.num_output = self.num_input

    def __interface_c__(self):
        self.interface_criterion = \
            "Crop(int axis,int offset,char* bottom, char* bottom_mode,char *top,char *name)"
        self.interface_c = "inferx_crop("
        self.interface_c += "{},{}".format(self.axis, self.offset)
        for index in range(len(self.bottom_layer)):
            self.interface_c += ",\"{}\"".format(self.bottom_layer[index].top)
        #self.interface_c += ",%d,bottom_vector" % len(self.bottom_layer)
        self.interface_c += ",\"{}\"".format(self.top)
        self.interface_c += ",\"{}\"".format(self.name)
        self.interface_c += ",{}".format(Layer.modelstr)
        self.interface_c += ",{});".format(Layer.datastr)


class Eltwise(Layer):
    """Eltwise layer"""

    __eltwise_phases = ['PROD', 'SUM', 'MAX']
    __phases_decimal = ['coeff']
    __phases_binary = ['stable_prod_grad']
    def __init__(self, layer_string=None,net_name=None):
        Layer.__init__(self, layer_string,net_name)
        self.operation = 'SUM'
        self.stabel_prod_grad = 'true'
        self.coeff = [1,1]

        self.__init_eltwise__()
        self.__init_binary_param__(self.__phases_binary[0], default='true')
        self.__init_decimal_param__(self.__phases_decimal)
        self.__list_all_member__()

    def __init_eltwise__(self):
        for phase in self.__eltwise_phases:
            phase_list = self.layer_string.split(phase)
            phase_num = len(phase_list)
            if phase_num == 1:
                self.__debug_print__("Eltwise layer %s has no eltwise operations named %s." % (self.name, phase))
            elif phase_num == 2:
                self.operation = phase
                self.__debug_print__("Eltwise layer %s has eltwise operations %s." % (self.name, self.operation))
            else:
                self.__debug_print__("Eltwise layer %s layer method error." % self.name)

    def __calc_ioput__(self):
        self.num_input = self.bottom_layer[0].num_output
        self.num_output = self.num_input

    def __interface_c__(self):
        self.interface_criterion = \
            "Eltwise(int coeffs_num, float* coeffs,enum EltwiseOp operation," \
            "bool stabel_prod_grad,int bottom_num,char **bottoms,char *top, char *name)"
        self.interface_c = ""
        for index in range(len(self.coeff)):
            self.interface_c += "coeffs[%d]=%f; " % (index,self.coeff[index])
        self.interface_c += "\n\tinferx_eltwise("
        # for index in range(len(self.coeff)):
        #     self.interface_c += "{}".format(self.coeff[index])
        self.interface_c += "%d,coeffs" % len(self.coeff)
        self.interface_c += ",{}".format(self.operation)
        self.interface_c += ",{}".format(self.stabel_prod_grad)
        # for index in range(len(self.bottom_layer)):
        #     self.interface_c += ",\"{}\"".format(self.bottom_layer[index].top)
        self.interface_c += ",%d,bottom_vector" % len(self.bottom_layer)
        self.interface_c += ",\"{}\"".format(self.top)
        self.interface_c += ",\"{}\"".format(self.name)
        self.interface_c += ",{}".format(Layer.modelstr)
        self.interface_c += ",{});".format(Layer.datastr)


class ReLU(Layer):
    """ReLU layer"""

    def __init__(self, layer_string=None,net_name=None):
        Layer.__init__(self, layer_string,net_name)
        self.__list_all_member__()

    def __calc_ioput__(self):
        self.num_input = self.bottom_layer[0].num_output
        self.num_output = self.num_input

    def __interface_c__(self):
        self.interface_criterion = \
            "ReLU(char *bottom,char *top,char *name)"
        self.interface_c = "inferx_Rele("
        for index in range(len(self.bottom_layer)):
            self.interface_c += "\"{}\"".format(self.bottom_layer[index].top)
        self.interface_c += ",\"{}\"".format(self.top)
        self.interface_c += ",\"{}\"".format(self.name)
        self.interface_c += ",{}".format(Layer.modelstr)
        self.interface_c += ",{});".format(Layer.datastr)

class InnerProduct(Layer):
    """InnerProduct layer"""

    __phases_number = ['num_output', 'axis']
    __phases_binary = ['bias_term', 'transpose']
    def __init__(self, layer_string=None,net_name=None):
        Layer.__init__(self, layer_string,net_name)
        self.axis = 1
        self.bias_term = 'true'
        self.transpose = 'false'

        self.__init_binary_param__(self.__phases_binary[0], default='true')
        self.__init_binary_param__(self.__phases_binary[1], default='false')
        self.__init_number_param__(self.__phases_number)

    def __calc_ioput__(self):
        self.num_input = self.bottom_layer[0].num_output

    def __interface_c__(self):
        self.interface_criterion = \
            "InnerProduct(int num_input,int num_output,bool bias_term," \
            "bool transpose,char *bottom,char *top,char *name)"
        self.interface_c = "inferx_innerproduct("
        self.interface_c += "{},{}".format(self.num_input, self.num_output)
        self.interface_c += ",{}".format(self.bias_term)
        self.interface_c += ",{}".format(self.transpose)
        for index in range(len(self.bottom_layer)):
            self.interface_c += ",\"{}\"".format(self.bottom_layer[index].top)
        self.interface_c += ",\"{}\"".format(self.top)
        self.interface_c += ",\"{}\"".format(self.name)
        self.interface_c += ",{}".format(self.modelstr)
        self.interface_c += ",{});".format(Layer.datastr)

class ArgMax(Layer):
    """ArgMax layer"""

    __phases_number = ['top_k', 'axis']
    __phases_binary = ['out_max_val']
    def __init__(self, layer_string=None,net_name=None):
        Layer.__init__(self, layer_string,net_name)
        self.out_max_val = 'false'
        self.top_k = 1
        self.axis = None

        self.__init_binary_param__(self.__phases_binary[0], default='false')
        self.__init_number_param__(self.__phases_number)
        self.__list_all_member__()

    def __calc_ioput__(self):
        self.num_input = self.bottom_layer[0].num_output
        self.num_output = self.num_input

    def __interface_c__(self):
        self.interface_criterion = \
            "ArgMax(int top_k,int axis,bool out_max_val,char *bottom,char *top,char *name)"
        self.interface_c = "inferx_argmax("
        self.interface_c += "{},{}".format(self.top_k, self.axis)
        self.interface_c += ",{}".format(self.out_max_val)
        for index in range(len(self.bottom_layer)):
            self.interface_c += ",\"{}\"".format(self.bottom_layer[index].top)
        self.interface_c += ",\"{}\"".format(self.top)
        self.interface_c += ",\"{}\"".format(self.name)
        self.interface_c += ",{}".format(Layer.modelstr)
        self.interface_c += ",{});".format(Layer.datastr)


class BatchNorm(Layer):
    """BatchNorm layer"""

    __phases_decimal = ['moving_average_fraction', 'eps']
    __phases_binary = ['use_global_stats']
    def __init__(self, layer_string=None,net_name=None):
        Layer.__init__(self, layer_string,net_name)
        self.use_global_stats = 'true'
        self.moving_average_fraction = ['0.999']
        self.eps = ['1e-5']

        self.__init_decimal_param__(self.__phases_decimal)
        self.__init_binary_param__(self.__phases_binary[0], default='true')
        self.__list_all_member__()

    def __calc_ioput__(self):
        self.__debug_print__(self.name)
        self.num_input = self.bottom_layer[0].num_output
        self.num_output = self.num_input

    def __interface_c__(self):
        self.interface_criterion = \
            "BatchNorm(float moving_average_fraction,float eps," \
            "bool use_global_stats,char *bottom,char *top,char *name)"
        self.interface_c = "inferx_batchnorm("
        self.interface_c += "{},{}".format(self.moving_average_fraction[0],self.eps[0])
        self.interface_c += ",{}".format(self.use_global_stats)
        for index in range(len(self.bottom_layer)):
            self.interface_c += ",\"{}\"".format(self.bottom_layer[index].top)
        self.interface_c += ",\"{}\"".format(self.top)
        self.interface_c += ",\"{}\"".format(self.name)
        self.interface_c += ",{}".format(Layer.modelstr)
        self.interface_c += ",{});".format(Layer.datastr)


class Concat(Layer):
    """Concat layer"""

    __phases_number = ['axis', 'concat_dim']
    def __init__(self, layer_string=None,net_name=None):
        Layer.__init__(self, layer_string,net_name)
        self.axis = 1
        self.concat_dim = 1
        self.__init_number_param__(self.__phases_number)
        self.__list_all_member__()

    def __calc_ioput__(self):
        self.num_input = []
        self.num_output = 0
        for bottom in self.bottom_layer:
            self.num_input.append(bottom.num_output)
        for input in self.num_input:
            self.num_output += int(input)

    def __interface_c__(self):
        self.interface_criterion = \
            "Concat(int num_output,int axis,int concat_dim," \
            "int bottom_num,char **bottoms,char *top,char *name)"
        self.interface_c = "inferx_concat("
        self.interface_c += "{},".format(self.num_output)
        self.interface_c += "{},{}".format(self.axis,self.concat_dim)
        # for index in range(len(self.bottom_layer)):
        #     self.interface_c += ",\"{}\"".format(self.bottom_layer[index].top)
        self.interface_c += ",%d,bottom_vector"%len(self.bottom_layer)
        self.interface_c += ",\"{}\"".format(self.top)
        self.interface_c += ",\"{}\"".format(self.name)
        self.interface_c += ",{}".format(Layer.modelstr)
        self.interface_c += ",{});".format(Layer.datastr)

class Scale(Layer):
    """Scale layer"""

    __phases_number = ['axis', 'num_axes']
    __phases_binary = ['bias_term']
    def __init__(self, layer_string=None,net_name=None):
        Layer.__init__(self, layer_string,net_name)
        self.axis = 1
        self.num_axes = 1
        self.bias_term = 'false'
        self.__init_number_param__(self.__phases_number)
        self.__init_binary_param__(self.__phases_binary[0], default='false')
        self.__list_all_member__()

    def __calc_ioput__(self):
        self.num_input = self.bottom_layer[0].num_output
        self.num_output = self.num_input

    def __interface_c__(self):
        self.interface_criterion = \
            "Scale(int axis,int num_axes,bool bias_term,char *bottom,char *top,char *name)"
        self.interface_c = "inferx_scale("
        self.interface_c += "{},{}".format(self.axis, self.num_axes)
        self.interface_c += ",{}".format(self.bias_term)
        for index in range(len(self.bottom_layer)):
            self.interface_c += ",\"{}\"".format(self.bottom_layer[index].top)
        self.interface_c += ",\"{}\"".format(self.top)
        self.interface_c += ",\"{}\"".format(self.name)
        self.interface_c += ",{}".format(Layer.modelstr)
        self.interface_c += ",{});".format(Layer.datastr)

class Sigmoid(Layer):
    """Sigmoid layer"""

    def __init__(self, layer_string=None,net_name=None):
        Layer.__init__(self, layer_string,net_name)
        self.__list_all_member__()

    def __calc_ioput__(self):
        self.num_input = self.bottom_layer[0].num_output
        self.num_output = self.num_input

    def __interface_c__(self):
        self.interface_criterion = \
            "Sigmoid(char *bottom,char *top,char *name)"
        self.interface_c = "inferx_sigmoid("
        for index in range(len(self.bottom_layer)):
            self.interface_c += "\"{}\"".format(self.bottom_layer[index].top)
        self.interface_c += ",\"{}\"".format(self.top)
        self.interface_c += ",\"{}\"".format(self.name)
        self.interface_c += ",{}".format(Layer.modelstr)
        self.interface_c += ",{});".format(Layer.datastr)

class Softmax(Layer):
    """Softmax layer"""

    __phases_number = ['axis']
    def __init__(self, layer_string=None,net_name=None):
        Layer.__init__(self, layer_string,net_name)
        self.axis = 1
        self.__init_number_param__(self.__phases_number)
        self.__list_all_member__()

    def __calc_ioput__(self):
        self.num_input = self.bottom_layer[0].num_output
        self.num_output = self.num_input

    def __interface_c__(self):
        self.interface_criterion = \
            "Softmax(int axis,char *bottom,char *top,char *name)"
        self.interface_c = "inferx_softmax("
        self.interface_c += "{}".format(self.axis)
        for index in range(len(self.bottom_layer)):
            self.interface_c += ",\"{}\"".format(self.bottom_layer[index].top)
        self.interface_c += ",\"{}\"".format(self.top)
        self.interface_c += ",\"{}\"".format(self.name)
        self.interface_c += ",{}".format(Layer.modelstr)
        self.interface_c += ",{});".format(Layer.datastr)

class TanH(Layer):
    """TanH layer"""

    def __init__(self, layer_string=None,net_name=None):
        Layer.__init__(self, layer_string,net_name)
        self.__list_all_member__()

    def __calc_ioput__(self):
        self.num_input = self.bottom_layer[0].num_output
        self.num_output = self.num_input

    def __interface_c__(self):
        self.interface_criterion = \
            "TanH(char *bottom,char *top,char *name)"
        self.interface_c = "inferx_tanh("
        for index in range(len(self.bottom_layer)):
            self.interface_c += "\"{}\"".format(self.bottom_layer[index].top)
        self.interface_c += ",\"{}\"".format(self.top)
        self.interface_c += ",\"{}\"".format(self.name)
        self.interface_c += ",{}".format(Layer.modelstr)
        self.interface_c += ",{});".format(Layer.datastr)

class LRN(Layer):
    """LRN layer"""

    __phases_decimal = ['alpha', 'beta', 'k']
    __phases_number = ['local_size']
    def __init__(self, layer_string=None,net_name=None):
        Layer.__init__(self, layer_string,net_name)
        self.local_size = 5
        self.alpha = [1.0]
        self.beta = [0.75]
        self.k = [1.0]
        self.__init_number_param__(self.__phases_number)
        self.__init_decimal_param__(self.__phases_decimal)

    def __calc_ioput__(self):
        self.num_input = self.bottom_layer[0].num_output
        self.num_output = self.num_input

    def __interface_c__(self):
        self.interface_criterion = \
            "LRN(int local_size,float alpha,float beta,float k,char *bottom,char *top,char *name)"
        self.interface_c = "inferx_LRN("
        self.interface_c += "{},{},{},{}".format(self.local_size, self.alpha[0], self.beta[0],self.k[0])
        for index in range(len(self.bottom_layer)):
            self.interface_c += ",\"{}\"".format(self.bottom_layer[index].top)
        self.interface_c += ",\"{}\"".format(self.top)
        self.interface_c += ",\"{}\"".format(self.name)
        self.interface_c += ",{}".format(Layer.modelstr)
        self.interface_c += ",{});".format(Layer.datastr)

class Dropout(Layer):
    """Dropout layer"""

    def __init__(self, layer_string,net_name=None):
        Layer.__init__(self, layer_string,net_name)

class Reshape(Layer):
    """Reshape layer"""
    __phases_number = ['dim']
    
    def  __init__(self, layer_string,net_name=None):
         Layer.__init__(self, layer_string,net_name)
         self.dim= []
	
         self.__init_dim__()	
	 self.__list_all_member__()

         self.batch_size = self.dim[0]
	 self.channels = self.dim[1]
         self.height = self.dim[2]
         self.width = self.dim[3]
	
    def __init_dim__(self):
        phase_list = self.layer_string.split('dim:')
        phase_num = len(phase_list)
        if phase_num == 1:
            self.__debug_print__("Input layer %s has no input dims" % self.name, printout=True)
        elif phase_num >= 2:
            print("phase_num".format(phase_num))
            print(phase_list[1])
            for index in range(1, phase_num):
                self.dim.extend(self.__find_all_num__(phase_list[index]))

    def __calc_ioput__(self):
        self.num_input = self.bottom_layer[0].num_output
	self.num_output = self.channels

    def __interface_c__(self):
        self.interface_criterion = \
            "inferx_reshape(int batch_size, int channels, int height, int weidth char *bottom, char *top char *name)"
        self.interface_c = "Reshape("
        self.interface_c +="{},{},{},{}".\
            format(self.batch_size,self.channels,self.height,self.width)
        self.interface_c +=",\"{}\",\"{}\",\"{}\",{},{});".format(self.bottom_layer[0].top,self.top,self.name,Layer.modelstr,Layer.datastr)




class Net(object):
    """Convert caffe net protobuf file to inferxlite net.c file"""

    def __init__(self, proto=None,is_merge_bn=False,is_merge_relu=False):
        self.__loaded = False
        self.__proto = proto
	
	self.__merge_bn=is_merge_bn
	self.__merge_ReLU=is_merge_relu
        self.__name = None
        self.__layers_string = None
        self.__layers = []
        self.__layernum = None
        self.__log = []
        self.__net = ""
        self.__cfile = []

        self.__read_proto__()
        self.__init_layers_()
        self.__link_layers__()

        self.__all_layers_type = self.__all_layers_type__()
        self.__write_c_format__(annotation=True)

    def __update_log__(self, log, printout=False):
        """Print log from here"""
        if printout:
            print(log)
        self.__log.append(log)

    def __update_line__(self, line, outlines, printout=False):
        """Print line from here"""
        if printout:
            print(line)
        outlines.append(line)

    def __read_proto__(self):
        """Read caffe net protobuf file"""
        try:
            net_lines = open(self.__proto, "r").readlines()
            for line in net_lines:
                if not hasannotation(line):
                    self.__net += line
                else:
                    self.__net += dropannotation(line)
        except IOError:
            self.__update_log__("IOError file {} not opened." % self.__proto)
            return
        self.__layers_string = self.__net.split('layer {')
        if not len(self.__layers_string[0].split("name:")) == 1:
            if not len(self.__layers_string[0].split("name:")[1].split('\"')) == 1:
                self.__name = self.__layers_string[0].split("name:")[1].split('\"')[1]
        if self.__name == None:
            self.__name = input("Please input the net name using \"name\" format:")
        self.__name = cformatparam(self.__name)
        self.__update_log__("Net has been loaded successfully.")
        self.__loaded = True

    def __init_layers_(self):
        if not self.__loaded:
            self.__update_log__("Net not loaded, please check your net proto file.")
        else:
            if len(self.__layers_string[0].split("dim:")) >= 2:
                self.__layers.append(LayerFactory(layer_string=self.__layers_string[0],net_name=self.__name).layer)
            for layer_string in self.__layers_string[1:]:
                self.__layers.append(LayerFactory(layer_string=layer_string,net_name=self.__name).layer)
            self.__update_log__("Layers has initialized successfully.")

    def __link_layers__(self):
        for index_i in range(len(self.__layers)):
            if self.__layers[index_i].bottom == None:
                self.__layers[index_i].__calc_ioput__()
                self.__layers[index_i].__interface_c__()
                continue
            bottom_num = len(self.__layers[index_i].bottom)
            self.__layers[index_i].bottom_layer = []
            for index_ib in range(bottom_num):
                for index_j in range(index_i):
                    if self.__layers[index_i].bottom[index_ib] == self.__layers[index_j].top:
                        self.__layers[index_i].bottom_layer.append(self.__layers[index_j])
                        break
            self.__layers[index_i].__calc_ioput__()
            if(self.__merge_bn and self.__layers[index_i].type =="Convolution"):
                self.__layers[index_i].__set_merge_bn__()
            self.__layers[index_i].__interface_c__()

    def __all_layers_type__(self):
        types = []
        for index in range(len(self.__layers)):
            type = {"{}".format(self.__layers[index].type):self.__layers[index].interface_criterion}
            if not type in types:
                types.append(type)
        return types

    def __write_annotations__(self):
        line = "/*\n" \
               "\tThis file is generated by net_compiler.py.\n" \
               "\tThe use of included functions list as follows:\n"

        for type in self.__all_layers_type:
            for key in type:
                if not type[key] == None:
                    line += '\n\t' + key + ':\n'
                    line += '\t' + type[key] + '\n'
        line += '*/\n\n'
        self.__update_line__(line, self.__cfile)

    def __write_c_format__(self, annotation=False):
        outf = open("{}.c".format(self.__name), 'w+')
        if annotation:
            self.__write_annotations__()
        lines = "#include \"inferxlite_common.h\"\n"
        lines += "#include \"interface.h\"\n"
        #lines += "#include \"caffe.h\"\n\n"
        lines += "void " + self.__name + "(char * path, char * model, char * data_c, void * pdata)\n{\n"
        max_bottom = 1
        max_len_coeff = 1
        for index in range(len(self.__layers)):
            if self.__layers[index].type == "Eltwise":
                if len(self.__layers[index].coeff) > max_len_coeff:
                    max_len_coeff = len(self.__layers[index].coeff)
            if not self.__layers[index].bottom == None:
                if len(self.__layers[index].bottom) > max_bottom:
                    max_bottom = len(self.__layers[index].bottom)
        if max_len_coeff > 1:
            lines += "\tfloat coeffs[%d];\n" % max_len_coeff
        if max_bottom > 1:
            lines += "\tchar* bottom_vector[%d];\n\n" % max_bottom
	##add function
	lines += "\tlong nchw[4];\n"
	lines += "\tchar data[1000];\n"
	lines += "\tinferx_parse_str(data_c, nchw, data);\n"
        lines += "\tinferx_set_init_var(&weightHasLoad, &dataHasInit, model, data);\n"
        lines += "\tinferx_var_add_init(model);\n"
        lines += "\tinfer_var_add_init(data);\n"
	lines += "\n"
        #lines += "\tinsertModelFunc"+ "(\""+self.__name+"\","+self.__name+");\n"
        for index in range(len(self.__layers)):
            if(self.__layers[index].interface_c == None):
                self.__update_log__("Ignore layer {}.".format(self.__layers[index].name))
                continue
            if not self.__layers[index].bottom == None:
                if self.__layers[index].type == "Eltwise" or self.__layers[index].type == "Concat":
                    for bottom_i in range(len(self.__layers[index].bottom)):
                        lines += "\tbottom_vector[%d] = \"%s\";" % (bottom_i,self.__layers[index].bottom[bottom_i])
                    lines += "\n"
            """merge BatchNorm"""
            if(self.__merge_bn==True and index+1<len(self.__layers) and self.__layers[index].type =="Convolution" and self.__layers[index+1].type =="BatchNorm"):
                self.__layers[index].__set_bias_term__()
            if(self.__merge_bn==True and self.__layers[index-1].type == "Convolution" and self.__layers[index].type =="BatchNorm"):
                continue
            if(self.__merge_bn==True and self.__layers[index- 2].type == "Convolution" and  self.__layers[index-1].type == "BatchNorm" and self.__layers[index].type =="Scale"):
                continue
            """merge ReLU """
            if(self.__layers[index].type == "Convolution"):
		if((self.__merge_ReLU and index+1<len(self.__layers) and self.__layers[index+1].type == "ReLU") or self.__merge_ReLU and (index+3<len(self.__layers) and self.__layers[index+3].type =="ReLU")):
		    lines += "\t{} 1);\n".format(self.__layers[index].interface_c)
                    print("index %d",index)
                    print("delete relu")
                else:
                    lines += "\t{} 0);\n".format(self.__layers[index].interface_c)
                    #print("do not delete relu")
                continue
            if(self.__merge_ReLU and self.__layers[index].type == "ReLU" and (self.__layers[index-1].type == "Convolution" or  self.__layers[index-3].type =="Convolution")):
                continue

            lines += "\t{}\n".format(self.__layers[index].interface_c)
        lines += "\n\t//inferx_sort_data(\"{}\",{});\n".format(self.__layers[-1].top,"data")
        lines += "\t//inferx_print_data(\"{}\",{});\n".format(self.__layers[-1].top,"data")
        #lines += "\tsaveData(\"{}\");\n\n".format(self.__layers[-1].top)
        lines += "\tinferx_finalize(\"{}\");\n".format(self.__name)
        lines += "\n\treturn;\n}"
        self.__update_line__(lines, self.__cfile)
        outf.writelines(self.__cfile)
       
        outf = open("{}.h".format(self.__name), 'w+')
        line = "extern void {}();".format(self.__name)
        outf.writelines(line)

def get_user_paras():
    try:
        opt = OptionParser()
        opt.add_option('-f','--input_file',
                       dest='input_file',
                       type = str,
                       help='input *.prototxt file')
        opt.add_option('-m','--merge_bn',
                       action='store_true',
                       dest='is_merge_bn',
                       default=False,
                       help = "merge batchnorm layer with convolution")
        opt.add_option('-n','--no_merge_bn',
                       action='store_false',
                       dest='is_merge_bn',
                       default=False,
                       help = "do not merge batchnorm layer with convolution")
        opt.add_option('-r','--merge_relu',
                       action='store_true',
                       dest='is_merge_relu',
                       default=False,
                       help = "merge relu layer with convolution")
        opt.add_option('-p','--no_merge_relu',
                       action='store_false',
                       dest='is_merge_relu',
                       default=False,
                       help = "do not merge relu layer with convolution")
        (options,args) = opt.parse_args()
        is_valid_paras =True
        error_messages = []
        input_file = options.input_file
        is_merge_bn = options.is_merge_bn
        is_merge_relu = options.is_merge_relu
        if not input_file:
            error_messages.append("input file must be set;")
            is_valid_paras = False
        if is_valid_paras:
            user_paras={"input_file":input_file,"is_merge_bn":is_merge_bn,"is_merge_relu":is_merge_relu}
            return user_paras
        else:
            for error_message in error_messages:
                print(error_message)
            opt.print_help()
            return None
  
    except Exception as ex:
        print("exception :{0}".format(str(ex)))
        return None






if __name__ == "__main__":
    user_paras = get_user_paras()
    if user_paras is None:
        sys.exit(-1) 
    net = Net(user_paras["input_file"],user_paras["is_merge_bn"],user_paras["is_merge_relu"])
    #if not len(sys.argv) == 1:
        #net = Net(str(sys.argv[1]))
    #else:
        #net = Net("deploy.prototxt")

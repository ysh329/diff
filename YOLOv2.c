/*
	This file is generated by net_compiler.py.
	The use of included functions list as follows:

	Input:
	Input(int dim1,int dim2,int dim3,int dim4,char *top,char *name)

	Convolution:
	Convolution(int num_input,int num_output,int kernel_h,int kernel_w,int stride_h,int stride_w,int pad_h,int pad_w,int group,int dilation,int axis,bool bias_term,bool force_nd_im2col,char *bottom,char *top, char *name)

	BatchNorm:
	BatchNorm(float moving_average_fraction,float eps,bool use_global_stats,char *bottom,char *top,char *name)

	Scale:
	Scale(int axis,int num_axes,bool bias_term,char *bottom,char *top,char *name)

	ReLU:
	ReLU(char *bottom,char *top,char *name)

	Pooling:
	Pooling(int kernel_h,int kernel_w,int stride_h,int stride_w,int pad_h,int pad_w,enum PoolMethod pool,char *bottom,char *top,char *name)

	Reshape:
	Reshape(int batch_size, int channels, int height, int weidth char *bottom, char *top char *name)

	Concat:
	Concat(int num_output,int axis,int concat_dim,int bottom_num,char **bottoms,char *top,char *name)
*/

#include "inferxlite_common.h"
#include "interface.h"
void YOLOv2(char * path, char * model, char * data_c, void * pdata)
{
	char* bottom_vector[2];

	long nchw[4];
	char data[1000];
	parseStr(data_c, nchw, data);
	setInitVar(&weightHasLoad, &dataHasInit, model, data);
	varAddInit(model);
	varAddInit(data);

	Input(nchw,pdata,"data_data","data",model,data);
	Convolution(3,32,3,3,1,1,1,1,1,1,1,false,false,"data_data","layer1-conv_data","layer1-conv",model,data);
	BatchNorm(0.999,1e-5,true,"layer1-conv_data","layer1-conv_data","layer1-bn",model,data);
	Scale(1,1,true,"layer1-conv_data","layer1-conv_data","layer1-scale",model,data);
	ReLU("layer1-conv_data","layer1-conv_data","layer1-act",model,data);
	Pooling(2,2,2,2,0,0,MAX,"layer1-conv_data","layer2-maxpool_data","layer2-maxpool",model,data);
	Convolution(32,64,3,3,1,1,1,1,1,1,1,false,false,"layer2-maxpool_data","layer3-conv_data","layer3-conv",model,data);
	BatchNorm(0.999,1e-5,true,"layer3-conv_data","layer3-conv_data","layer3-bn",model,data);
	Scale(1,1,true,"layer3-conv_data","layer3-conv_data","layer3-scale",model,data);
	ReLU("layer3-conv_data","layer3-conv_data","layer3-act",model,data);
	Pooling(2,2,2,2,0,0,MAX,"layer3-conv_data","layer4-maxpool_data","layer4-maxpool",model,data);
	Convolution(64,128,3,3,1,1,1,1,1,1,1,false,false,"layer4-maxpool_data","layer5-conv_data","layer5-conv",model,data);
	BatchNorm(0.999,1e-5,true,"layer5-conv_data","layer5-conv_data","layer5-bn",model,data);
	Scale(1,1,true,"layer5-conv_data","layer5-conv_data","layer5-scale",model,data);
	ReLU("layer5-conv_data","layer5-conv_data","layer5-act",model,data);
	Convolution(128,64,1,1,1,1,0,0,1,1,1,false,false,"layer5-conv_data","layer6-conv_data","layer6-conv",model,data);
	BatchNorm(0.999,1e-5,true,"layer6-conv_data","layer6-conv_data","layer6-bn",model,data);
	Scale(1,1,true,"layer6-conv_data","layer6-conv_data","layer6-scale",model,data);
	ReLU("layer6-conv_data","layer6-conv_data","layer6-act",model,data);
	Convolution(64,128,3,3,1,1,1,1,1,1,1,false,false,"layer6-conv_data","layer7-conv_data","layer7-conv",model,data);
	BatchNorm(0.999,1e-5,true,"layer7-conv_data","layer7-conv_data","layer7-bn",model,data);
	Scale(1,1,true,"layer7-conv_data","layer7-conv_data","layer7-scale",model,data);
	ReLU("layer7-conv_data","layer7-conv_data","layer7-act",model,data);
	Pooling(2,2,2,2,0,0,MAX,"layer7-conv_data","layer8-maxpool_data","layer8-maxpool",model,data);
	Convolution(128,256,3,3,1,1,1,1,1,1,1,false,false,"layer8-maxpool_data","layer9-conv_data","layer9-conv",model,data);
	BatchNorm(0.999,1e-5,true,"layer9-conv_data","layer9-conv_data","layer9-bn",model,data);
	Scale(1,1,true,"layer9-conv_data","layer9-conv_data","layer9-scale",model,data);
	ReLU("layer9-conv_data","layer9-conv_data","layer9-act",model,data);
	Convolution(256,128,1,1,1,1,0,0,1,1,1,false,false,"layer9-conv_data","layer10-conv_data","layer10-conv",model,data);
	BatchNorm(0.999,1e-5,true,"layer10-conv_data","layer10-conv_data","layer10-bn",model,data);
	Scale(1,1,true,"layer10-conv_data","layer10-conv_data","layer10-scale",model,data);
	ReLU("layer10-conv_data","layer10-conv_data","layer10-act",model,data);
	Convolution(128,256,3,3,1,1,1,1,1,1,1,false,false,"layer10-conv_data","layer11-conv_data","layer11-conv",model,data);
	BatchNorm(0.999,1e-5,true,"layer11-conv_data","layer11-conv_data","layer11-bn",model,data);
	Scale(1,1,true,"layer11-conv_data","layer11-conv_data","layer11-scale",model,data);
	ReLU("layer11-conv_data","layer11-conv_data","layer11-act",model,data);
	Pooling(2,2,2,2,0,0,MAX,"layer11-conv_data","layer12-maxpool_data","layer12-maxpool",model,data);
	Convolution(256,512,3,3,1,1,1,1,1,1,1,false,false,"layer12-maxpool_data","layer13-conv_data","layer13-conv",model,data);
	BatchNorm(0.999,1e-5,true,"layer13-conv_data","layer13-conv_data","layer13-bn",model,data);
	Scale(1,1,true,"layer13-conv_data","layer13-conv_data","layer13-scale",model,data);
	ReLU("layer13-conv_data","layer13-conv_data","layer13-act",model,data);
	Convolution(512,256,1,1,1,1,0,0,1,1,1,false,false,"layer13-conv_data","layer14-conv_data","layer14-conv",model,data);
	BatchNorm(0.999,1e-5,true,"layer14-conv_data","layer14-conv_data","layer14-bn",model,data);
	Scale(1,1,true,"layer14-conv_data","layer14-conv_data","layer14-scale",model,data);
	ReLU("layer14-conv_data","layer14-conv_data","layer14-act",model,data);
	Convolution(256,512,3,3,1,1,1,1,1,1,1,false,false,"layer14-conv_data","layer15-conv_data","layer15-conv",model,data);
	BatchNorm(0.999,1e-5,true,"layer15-conv_data","layer15-conv_data","layer15-bn",model,data);
	Scale(1,1,true,"layer15-conv_data","layer15-conv_data","layer15-scale",model,data);
	ReLU("layer15-conv_data","layer15-conv_data","layer15-act",model,data);
	Convolution(512,256,1,1,1,1,0,0,1,1,1,false,false,"layer15-conv_data","layer16-conv_data","layer16-conv",model,data);
	BatchNorm(0.999,1e-5,true,"layer16-conv_data","layer16-conv_data","layer16-bn",model,data);
	Scale(1,1,true,"layer16-conv_data","layer16-conv_data","layer16-scale",model,data);
	ReLU("layer16-conv_data","layer16-conv_data","layer16-act",model,data);
	Convolution(256,512,3,3,1,1,1,1,1,1,1,false,false,"layer16-conv_data","layer17-conv_data","layer17-conv",model,data);
	BatchNorm(0.999,1e-5,true,"layer17-conv_data","layer17-conv_data","layer17-bn",model,data);
	Scale(1,1,true,"layer17-conv_data","layer17-conv_data","layer17-scale",model,data);
	ReLU("layer17-conv_data","layer17-conv_data","layer17-act",model,data);
	Pooling(2,2,2,2,0,0,MAX,"layer17-conv_data","layer18-maxpool_data","layer18-maxpool",model,data);
	Convolution(512,1024,3,3,1,1,1,1,1,1,1,false,false,"layer18-maxpool_data","layer19-conv_data","layer19-conv",model,data);
	BatchNorm(0.999,1e-5,true,"layer19-conv_data","layer19-conv_data","layer19-bn",model,data);
	Scale(1,1,true,"layer19-conv_data","layer19-conv_data","layer19-scale",model,data);
	ReLU("layer19-conv_data","layer19-conv_data","layer19-act",model,data);
	Convolution(1024,512,1,1,1,1,0,0,1,1,1,false,false,"layer19-conv_data","layer20-conv_data","layer20-conv",model,data);
	BatchNorm(0.999,1e-5,true,"layer20-conv_data","layer20-conv_data","layer20-bn",model,data);
	Scale(1,1,true,"layer20-conv_data","layer20-conv_data","layer20-scale",model,data);
	ReLU("layer20-conv_data","layer20-conv_data","layer20-act",model,data);
	Convolution(512,1024,3,3,1,1,1,1,1,1,1,false,false,"layer20-conv_data","layer21-conv_data","layer21-conv",model,data);
	BatchNorm(0.999,1e-5,true,"layer21-conv_data","layer21-conv_data","layer21-bn",model,data);
	Scale(1,1,true,"layer21-conv_data","layer21-conv_data","layer21-scale",model,data);
	ReLU("layer21-conv_data","layer21-conv_data","layer21-act",model,data);
	Convolution(1024,512,1,1,1,1,0,0,1,1,1,false,false,"layer21-conv_data","layer22-conv_data","layer22-conv",model,data);
	BatchNorm(0.999,1e-5,true,"layer22-conv_data","layer22-conv_data","layer22-bn",model,data);
	Scale(1,1,true,"layer22-conv_data","layer22-conv_data","layer22-scale",model,data);
	ReLU("layer22-conv_data","layer22-conv_data","layer22-act",model,data);
	Convolution(512,1024,3,3,1,1,1,1,1,1,1,false,false,"layer22-conv_data","layer23-conv_data","layer23-conv",model,data);
	BatchNorm(0.999,1e-5,true,"layer23-conv_data","layer23-conv_data","layer23-bn",model,data);
	Scale(1,1,true,"layer23-conv_data","layer23-conv_data","layer23-scale",model,data);
	ReLU("layer23-conv_data","layer23-conv_data","layer23-act",model,data);
	Convolution(1024,1024,3,3,1,1,1,1,1,1,1,false,false,"layer23-conv_data","layer24-conv_data","layer24-conv",model,data);
	BatchNorm(0.999,1e-5,true,"layer24-conv_data","layer24-conv_data","layer24-bn",model,data);
	Scale(1,1,true,"layer24-conv_data","layer24-conv_data","layer24-scale",model,data);
	ReLU("layer24-conv_data","layer24-conv_data","layer24-act",model,data);
	Convolution(1024,1024,3,3,1,1,1,1,1,1,1,false,false,"layer24-conv_data","layer25-conv_data","layer25-conv",model,data);
	BatchNorm(0.999,1e-5,true,"layer25-conv_data","layer25-conv_data","layer25-bn",model,data);
	Scale(1,1,true,"layer25-conv_data","layer25-conv_data","layer25-scale",model,data);
	ReLU("layer25-conv_data","layer25-conv_data","layer25-act",model,data);
	Reorg(0,2048,9,9,"layer17-conv_data","layer27-reshape_data","layer27-reshape",model,data);
	bottom_vector[0] = "layer27-reshape_data";	bottom_vector[1] = "layer25-conv_data";
	//Concat(1536,1,1,2,bottom_vector,"layer28-concat_data","layer28-concat",model,data);
        Concat(3072,1,1,2,bottom_vector,"layer28-concat_data","layer28-concat",model,data);



	//Convolution(1536,1024,3,3,1,1,1,1,1,1,1,false,false,"layer28-concat_data","layer29-conv_data","layer29-conv",model,data);
        Convolution(3072,1024,3,3,1,1,1,1,1,1,1,false,false,"layer28-concat_data","layer29-conv_data","layer29-conv",model,data);

	BatchNorm(0.999,1e-5,true,"layer29-conv_data","layer29-conv_data","layer29-bn",model,data);
	Scale(1,1,true,"layer29-conv_data","layer29-conv_data","layer29-scale",model,data);
	ReLU("layer29-conv_data","layer29-conv_data","layer29-act",model,data);

	Convolution(1024,75,1,1,1,1,0,0,1,1,1,true,false,"layer29-conv_data","layer30-conv_data","layer30-conv",model,data);

        //printData_v2("layer30-conv_data", );
        
	//sortData("layer30-conv_data",data);
        printf("===");
        getData("layer30-conv_data",data);

	Finalize("YOLOv2");

	return;
}

//
//  Caffe2.m
//  MyCaffe
//
//  Created by quanhai on 2017/5/9.
//  Copyright © 2017年 Quanhai. All rights reserved.
//

#import "Caffe2.h"
#import "UIImage+ReSize.h"

#import "Image_classes.h"

//# import "caffe2/core/context.h"
//# import "caffe2/core/operator.h"
//# import "Caffe2.h"
// Caffe2 Headers
#include "caffe2/core/flags.h"
#include "caffe2/core/init.h"
#include "caffe2/core/predictor.h"
#include "caffe2/utils/proto_utils.h"
// OpenCV
#define HAVE_OPENCV_IMGPROC
#import <opencv2/opencv.hpp>


void UIImageToMat(const UIImage* image,
                  cv::Mat& m, bool alphaExist) {
    CGColorSpaceRef colorSpace = CGImageGetColorSpace(image.CGImage);
    CGFloat cols = image.size.width, rows = image.size.height;
    CGContextRef contextRef;
    CGBitmapInfo bitmapInfo = kCGImageAlphaPremultipliedLast;
    if (CGColorSpaceGetModel(colorSpace) == 0)
    {
        m.create(rows, cols, CV_8UC1); // 8 bits per component, 1 channel
        bitmapInfo = kCGImageAlphaNone;
        if (!alphaExist)
            bitmapInfo = kCGImageAlphaNone;
        contextRef = CGBitmapContextCreate(m.data, m.cols, m.rows, 8,
                                           m.step[0], colorSpace,
                                           bitmapInfo);
    }
    else
    {
        m.create(rows, cols, CV_8UC4); // 8 bits per component, 4 channels
        if (!alphaExist)
            bitmapInfo = kCGImageAlphaNoneSkipLast |
            kCGBitmapByteOrderDefault;
        contextRef = CGBitmapContextCreate(m.data, m.cols, m.rows, 8,
                                           m.step[0], colorSpace,
                                           bitmapInfo);
    }
    CGContextDrawImage(contextRef, CGRectMake(0, 0, cols, rows),
                       image.CGImage);
    CGContextRelease(contextRef);
}




namespace caffe2 {
    
    void LoadPBFile(NSString *filePath, caffe2:: NetDef *net) {
        NSURL *netURL = [ NSURL fileURLWithPath:filePath];
        NSData *data = [ NSData dataWithContentsOfURL:netURL];
        const void *buffer = [data bytes];
        int len = (int)[data length];
        CAFFE_ENFORCE (net-> ParseFromArray (buffer, len));
    }

    Predictor *getPredictor( NSString *init_net_path, NSString *predict_net_path) {
        caffe2:: NetDef init_net, predict_net;
        LoadPBFile (init_net_path, &init_net);
        LoadPBFile (predict_net_path, &predict_net);
        auto predictor = new caffe2:: Predictor (init_net, predict_net);
        init_net.set_name( input_layer );
        predict_net.set_name( output_layer );
        return predictor;
    }
}



@implementation Caffe2
{
    caffe2:: Predictor *predictor;
}

- (instancetype) initWithDelegate:(id)delegate {
    self = [ super init ];
    if (self != nil ) {
        self.delegate = delegate;
        
        NSString *init_net_path = [NSBundle .mainBundle pathForResource:init_net_name ofType:init_net_type];
        NSString *predict_net_path = [ NSBundle .mainBundle pathForResource:predict_net_name ofType:predict_net_type];
        predictor = caffe2::getPredictor(init_net_path, predict_net_path);
    }
    return self;
}



- (void)anlizeImage:(UIImage *)inputImage withIdentifer:(NSString *)uniqueId
{
    self.uniqueId = uniqueId;
    inputImage = [inputImage scaledToSizeWithSameAspectRatio:CGSizeMake(predict_width, predict_height)];
//    inputImage  = [inputImage scaledToSize:CGSizeMake(predict_width, predict_height)];
    
    NSString *label = [self predictWithImage:inputImage];
    inputImage = nil;
    NSLog (@"Identified: %@" , label);
    // This is to allow us to use memory leak checks.
    google::protobuf:: ShutdownProtobufLibrary ();
}



- (NSString *)predictWithImage: (UIImage *)image {
    cv:: Mat src_img, bgra_img;
    UIImageToMat(image, src_img, false);
//    src_img = [self imageConverToMat:image];
    // needs to convert to BGRA because the image loaded from UIImage is in RGBA
    cv::cvtColor(src_img, bgra_img, CV_RGBA2RGB );
    size_t height = CGImageGetHeight(image.CGImage );
    size_t width = CGImageGetWidth(image.CGImage );
    caffe2:: TensorCPU input;
    // Reasonable dimensions to feed the predictor.
    const int predHeight = predict_height;
    const int predWidth = predict_width;
    const int crops = predict_crops;
    const int channels = image_channels;
    const int size = predHeight * predWidth;
    const float hscale = ((float)height) / predHeight;
    const float wscale = ((float)width) / predWidth;
    const float scale = std:: min(hscale, wscale);
//    const float scale = 1;
    std::vector<float> inputPlanar(crops * channels * predHeight * predWidth);
    // Scale down the input to a reasonable predictor size.
    CFAbsoluteTime start = CFAbsoluteTimeGetCurrent();
    
    for (auto i = 0; i < predHeight; ++i) {
        const int _i = (int) (scale * i);
//        printf( "+\n" );
        for (auto j = 0; j < predWidth; ++j) {
            const int _j = (int) (scale * j);
            
            float channel_0 = (float)bgra_img.data[(_i * width + _j) * 3 + 0];
            float channel_1 = (float)bgra_img.data[(_i * width + _j) * 3 + 1];
            float channel_2 = (float)(float)bgra_img.data[(_i * width + _j) * 3 + 2];
            
//            printf ("_i :%d ------_j : %d \n", _i, _j);
            
            inputPlanar[i * predWidth + j + 0 * size] = channel_0;
            inputPlanar[i * predWidth + j + 1 * size] = channel_1;
            inputPlanar[i * predWidth + j + 2 * size] = channel_2;
        }
    }
    CFAbsoluteTime end = CFAbsoluteTimeGetCurrent();
    NSLog(@"for 循环耗时：%.2f \n", end - start);
    
    input. Resize (std::vector<int>({crops, channels, predHeight, predWidth}));
    input. ShareExternalPointer (inputPlanar.data());
    caffe2:: Predictor :: TensorVector input_vec{&input};
    caffe2:: Predictor :: TensorVector output_vec;
    predictor->run(input_vec, &output_vec);
    float max_value = 0;
    int best_match_index = -1;
    for (auto output : output_vec) {
        for (auto i = 0; i < output->size(); ++i) {
            float val = output->template data<float>()[i];
            if (val > 0.001) {
                printf( "%i: %s : %f\n" , i, imagenet_classes[i], val);
                if (val>max_value) {
                    max_value = val;
                    best_match_index = i;
                }
            }
        }
    }
    NSString *label = [NSString stringWithUTF8String: imagenet_classes[best_match_index]];
    
    if([self.delegate respondsToSelector:@selector(caffe2DetectedImage: uniqueId: result:)]){
        [self.delegate caffe2DetectedImage:image uniqueId:self.uniqueId  result:label];
    }
    
    return label;
}

#pragma mark - 
-(cv::Mat)imageConverToMat:(UIImage *)image
{
    
    CGColorSpaceRef colorSpace = CGImageGetColorSpace(image.CGImage);
    CGFloat cols = image.size.width;
    CGFloat rows = image.size.height;
    
    cv::Mat cvMat(rows, cols, CV_8UC4); // 8 bits per component, 4 channels
    
    CGContextRef contextRef = CGBitmapContextCreate(cvMat.data,                 // Pointer to backing data
                                                    cols,                      // Width of bitmap
                                                    rows,                     // Height of bitmap
                                                    8,                          // Bits per component
                                                    cvMat.step[0],              // Bytes per row
                                                    colorSpace,                 // Colorspace
                                                    kCGImageAlphaNoneSkipLast |
                                                    kCGBitmapByteOrderDefault); // Bitmap info flags
    
    CGContextDrawImage(contextRef, CGRectMake(0, 0, cols, rows), image.CGImage);
    CGContextRelease(contextRef);
    
    return cvMat;
}



@end

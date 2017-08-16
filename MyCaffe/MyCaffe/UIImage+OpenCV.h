//
//  UIImage+OpenCV.h
//  MyCaffe
//
//  Created by quanhai on 2017/5/9.
//  Copyright © 2017年 Quanhai. All rights reserved.
//

#import <UIKit/UIKit.h>
#import "opencv2/opencv.hpp"



@interface UIImage (OpenCV)

+(UIImage *)imageWithCVMat:(const cv::Mat&)cvMat;
-(id)initWithCVMat:(const cv::Mat&)cvMat;
@property(nonatomic, readonly) cv::Mat CVMat;
@property(nonatomic, readonly) cv::Mat CVGrayscaleMat;


@end

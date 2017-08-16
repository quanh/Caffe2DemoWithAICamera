//
//  Caffe2.h
//  MyCaffe
//
//  Created by quanhai on 2017/5/9.
//  Copyright © 2017年 Quanhai. All rights reserved.
//

#import <Foundation/Foundation.h>
#import <UIKit/UIKit.h>

#ifdef __cplusplus
#import <iostream>
using namespace std;
#endif


@protocol Caffe2DetecteDelegate <NSObject>

- (void)caffe2DetectedImage:(UIImage *)image uniqueId:(NSString *)identifer result:(NSString *)result;
- (void)caffe2DectectedImage:(UIImage *)image results:(NSDictionary *)info;

@end



#ifdef __cplusplus
// models
static NSString *init_net_name = @"exec_net";
static NSString *init_net_type = @"pb";
static NSString *predict_net_name = @"predict_net";
static NSString *predict_net_type = @"pb";
// settings
const int predict_width = 256;
const int predict_height = 256;
const int predict_crops = 1;
const int image_channels = 3;

const std::string input_layer = "InitNet";
const std::string output_layer = "PredictNet";

#endif



@interface Caffe2 : NSObject

@property (nonatomic, copy) NSString *uniqueId;
@property (nonatomic, weak) id<Caffe2DetecteDelegate>delegate;

- (instancetype) initWithDelegate:(id)delegate;

- (void)anlizeImage:(UIImage *)image withIdentifer:(NSString *)uniqueId;


@end

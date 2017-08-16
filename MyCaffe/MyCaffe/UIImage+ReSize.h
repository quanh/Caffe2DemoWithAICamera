//
//  UIImage+ReSize.h
//  MyCaffe
//
//  Created by quanhai on 2017/5/11.
//  Copyright © 2017年 Quanhai. All rights reserved.
//

#import <UIKit/UIKit.h>


@interface UIImage (ReSize)

// use CoreGraphics
- (UIImage *)scaledToSizeWithSameAspectRatio:(CGSize)targetSize;
// use UIKit
- (UIImage*)scaledToSize:(CGSize)newSize;

@end

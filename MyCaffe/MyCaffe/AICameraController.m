//
//  AICameraController.m
//  MyCaffe
//
//  Created by quanhai on 2017/5/11.
//  Copyright © 2017年 Quanhai. All rights reserved.
//

#import "AICameraController.h"
#import <AVFoundation/AVFoundation.h>
#import "Caffe2.h"

@interface AICameraController ()
<
AVCaptureVideoDataOutputSampleBufferDelegate,
Caffe2DetecteDelegate
>

@property (nonatomic, weak) IBOutlet UIView *container;
@property (nonatomic, weak) IBOutlet UILabel *outputLabel;

@property (nonatomic, strong) Caffe2 *caffe2;
@property (nonatomic, strong) AVCaptureVideoPreviewLayer *previewLayer;
@property (nonatomic, strong) AVCaptureSession *session;
@property (nonatomic, strong) AVCaptureDeviceInput *input;
@property (nonatomic, strong) AVCaptureVideoDataOutput *output;
@property (nonatomic, assign) BOOL anlizeFinshed;

@end

@implementation AICameraController

- (AVCaptureSession *)session
{
    if (!_session) {
        _session = [[AVCaptureSession alloc] init];
    }
    return _session;
}
- (AVCaptureDeviceInput *)input
{
    if (!_input) {
//        AVCaptureDevice *device = [AVCaptureDevice defaultDeviceWithDeviceType:AVCaptureDeviceTypeBuiltInWideAngleCamera mediaType:AVMediaTypeVideo position:AVCaptureDevicePositionBack];
//        NSError *error;
//        _input = [[AVCaptureDeviceInput alloc] initWithDevice:device error:&error];
//        NSLog(@"%@",error.description);
        AVCaptureDevice *device = [AVCaptureDevice defaultDeviceWithMediaType:AVMediaTypeVideo];
        _input = [[AVCaptureDeviceInput alloc] initWithDevice:device error:nil];
    }
    return _input;
}
- (AVCaptureVideoDataOutput *)output
{
    if (!_output) {
        _output = [[AVCaptureVideoDataOutput alloc] init];
//        dispatch_queue_t outputQueue = dispatch_queue_create("videooutputQueue", DISPATCH_QUEUE_SERIAL);
//        [_output setSampleBufferDelegate:self queue:dispatch_get_main_queue()];
//        [_output setVideoSettings:@{(id)kCVPixelBufferPixelFormatTypeKey:@(kCVPixelFormatType_32BGRA)}];
    }
    return _output;
}
- (AVCaptureVideoPreviewLayer *)previewLayer
{
    if (!_previewLayer) {
        _previewLayer = [[AVCaptureVideoPreviewLayer alloc] initWithSession:self.session];
    }
    return _previewLayer;
}
- (Caffe2 *)caffe2
{
    if (!_caffe2) {
        _caffe2 = [[Caffe2 alloc] initWithDelegate:self];
    }
    return _caffe2;
}


+ (instancetype)getAICameraController
{
    AICameraController *cameraController = [[AICameraController alloc] initWithNibName:@"AICameraController" bundle:nil];
    cameraController.outputLabel.layer.cornerRadius = 8.f;
    cameraController.outputLabel.layer.masksToBounds = YES;
    
    return cameraController;
}


- (void)viewDidLoad {
    [super viewDidLoad];
    // Do any additional setup after loading the view from its nib.
    self.anlizeFinshed = YES;
    
    if ([self.session canAddInput:self.input]) {
        [self.session addInput:self.input];
    }
    
    NSDictionary *setttings = @{(id)kCVPixelBufferPixelFormatTypeKey : @(kCMPixelFormat_32BGRA)};
    [self.output setVideoSettings:setttings];
    dispatch_queue_t outputQueue = dispatch_queue_create("videooutputQueue", DISPATCH_QUEUE_SERIAL);
    [self.output setSampleBufferDelegate:self queue:outputQueue];
    
    if ([self.session canAddOutput:self.output]) {
        [self.session addOutput:self.output];
    }
    [[self.output connectionWithMediaType:AVMediaTypeVideo] setEnabled:YES];

    self.previewLayer.frame = [UIScreen mainScreen].bounds;
     [self.container.layer insertSublayer:self.previewLayer atIndex:0];
}


- (void)viewWillAppear:(BOOL)animated
{
    [super viewWillAppear:animated];
    
    if (![self.session isRunning]) {
        [self.session startRunning];
    }
}
- (void)viewWillDisappear:(BOOL)animated
{
    [super viewWillDisappear:animated];
    
    if ([self.session isRunning]) {
        [self.session stopRunning];
    }
}


- (void)caffe2DetectedImage:(UIImage *)image uniqueId:(NSString *)identifer result:(NSString *)result
{
    dispatch_async(dispatch_get_main_queue(), ^{
            self.outputLabel.text = result;
        self.anlizeFinshed = YES;
    });
}


#pragma mark - 
- (void)captureOutput:(AVCaptureOutput *)captureOutput didOutputSampleBuffer:(CMSampleBufferRef)sampleBuffer fromConnection:(AVCaptureConnection *)connection
{
//        UIImage *image = [self imageFromSampleBuffer:sampleBuffer];
        
        if(_anlizeFinshed)
        {
            self.anlizeFinshed = NO;

            UIImage *image = [self imageFromSampleBuffer:sampleBuffer];
            if (image) {
                 [self.caffe2 anlizeImage:image withIdentifer:nil];
            }else{
                self.anlizeFinshed = YES;
            }
        }
}


- (UIImage *)imageFromSampleBuffer:(CMSampleBufferRef) sampleBuffer
{
    
    // Get a CMSampleBuffer's Core Video image buffer for the media data
    CVImageBufferRef imageBuffer = CMSampleBufferGetImageBuffer(sampleBuffer);
    
    if (imageBuffer == nil) return nil;
    // Lock the base address of the pixel buffer
    CVPixelBufferLockBaseAddress(imageBuffer, 0);
    
    // Get the number of bytes per row for the pixel buffer
    void *baseAddress = CVPixelBufferGetBaseAddress(imageBuffer);
    
    // Get the number of bytes per row for the pixel buffer
    size_t bytesPerRow = CVPixelBufferGetBytesPerRow(imageBuffer);
    // Get the pixel buffer width and height
    size_t width = CVPixelBufferGetWidth(imageBuffer);
    size_t height = CVPixelBufferGetHeight(imageBuffer);
//    size_t width = 256;
//    size_t height = 256;
    
    // Create a device-dependent RGB color space
    CGColorSpaceRef colorSpace = CGColorSpaceCreateDeviceRGB();
    
    // Create a bitmap graphics context with the sample buffer data
    CGContextRef context = CGBitmapContextCreate(baseAddress, width, height, 8,
                                                 bytesPerRow, colorSpace, kCGBitmapByteOrder32Little | kCGImageAlphaPremultipliedFirst);

    // Create a Quartz image from the pixel data in the bitmap graphics context
    CGImageRef quartzImage = CGBitmapContextCreateImage(context);
    // Unlock the pixel buffer
    CVPixelBufferUnlockBaseAddress(imageBuffer,0);
    
    
    // Free up the context and color space
    CGContextRelease(context);
    CGColorSpaceRelease(colorSpace);
    
    // Create an image object from the Quartz image
    UIImage *image = [UIImage imageWithCGImage:quartzImage];
    
    // Release the Quartz image
    CGImageRelease(quartzImage);
    
    return (image);
}


#pragma mark - 
- (IBAction)dismissBack:(id)sender
{
    [self dismissViewControllerAnimated:YES completion:nil];
}








- (void)didReceiveMemoryWarning {
    [super didReceiveMemoryWarning];
    // Dispose of any resources that can be recreated.
}


@end

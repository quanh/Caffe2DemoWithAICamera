//
//  ViewController.m
//  MyCaffe
//
//  Created by quanhai on 2017/5/9.
//  Copyright © 2017年 Quanhai. All rights reserved.
//

#import "ViewController.h"
#import "AICameraController.h"
#import "Caffe2.h"

@interface ViewController ()
<
Caffe2DetecteDelegate,
UIImagePickerControllerDelegate,
UINavigationControllerDelegate
>
{
    CFAbsoluteTime startTime;
    CFAbsoluteTime endTime;
}
@property (nonatomic, weak) IBOutlet UIImageView *imageView;
@property (nonatomic, weak) IBOutlet UILabel *resultLabel;
@property (nonatomic, weak) IBOutlet UIButton *picButton;

@property (nonatomic, strong) Caffe2 *caffe2;

@end

@implementation ViewController

- (void)viewDidLoad {
    [super viewDidLoad];
    // Do any additional setup after loading the view, typically from a nib.
    
    _caffe2 = [[Caffe2 alloc] initWithDelegate:self];
    [self testCaffe2];
}

- (void)testCaffe2 {
    UIImage *test_img = [UIImage imageNamed:@"laduo"];
    self.imageView.image = test_img;
    [self detectImage:test_img];
}


- (IBAction)picPhoto:(id)sender
{
    UIImagePickerController *picker = [[UIImagePickerController alloc] init];
    picker.delegate = self;
    
    [self presentViewController:picker animated:YES completion:nil];
}

- (void)imagePickerController:(UIImagePickerController *)picker didFinishPickingMediaWithInfo:(NSDictionary<NSString *,id> *)info
{
    UIImage *image = info[UIImagePickerControllerOriginalImage];
    [self detectImage:image];
    [picker dismissViewControllerAnimated:YES completion:nil];
}

- (IBAction)goToAICamera:(id)sender
{
    AICameraController *camera = [AICameraController getAICameraController];
    
    [self presentViewController:camera animated:YES completion:nil];
}

#pragma mark - detect image

- (void)detectImage:(UIImage *)image
{
    startTime = CFAbsoluteTimeGetCurrent();
    [_caffe2 anlizeImage:image withIdentifer:nil];
}

#pragma mark - Caffe2DetectedDelegate
- (void)caffe2DetectedImage:(UIImage *)image uniqueId:(NSString *)identifer result:(NSString *)result
{
    endTime = CFAbsoluteTimeGetCurrent();
    CGFloat duration = (endTime - startTime);
    NSString *costTime = [NSString stringWithFormat:@"\n 使用时间：%.4f秒",duration];
    NSString *outputString = [result stringByAppendingString:costTime];
    
    self.imageView.image = image;
    self.resultLabel.text = outputString;
}




- (void)didReceiveMemoryWarning {
    [super didReceiveMemoryWarning];
    // Dispose of any resources that can be recreated.
}


@end

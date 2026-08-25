#import <AppKit/AppKit.h>
#import <Vision/Vision.h>

int main(int argc, const char *argv[]) {
    @autoreleasepool {
        if (argc != 2) return 2;

        NSString *imagePath = [NSString stringWithUTF8String:argv[1]];
        NSImage *image = [[NSImage alloc] initWithContentsOfFile:imagePath];
        if (!image) return 3;

        NSRect proposedRect = NSMakeRect(0, 0, image.size.width, image.size.height);
        CGImageRef cgImage = [image CGImageForProposedRect:&proposedRect context:nil hints:nil];
        if (!cgImage) return 4;

        VNRecognizeTextRequest *request = [[VNRecognizeTextRequest alloc] init];
        request.recognitionLevel = VNRequestTextRecognitionLevelAccurate;
        request.usesLanguageCorrection = NO;
        request.recognitionLanguages = @[@"en-US"];

        NSError *error = nil;
        VNImageRequestHandler *handler = [[VNImageRequestHandler alloc] initWithCGImage:cgImage options:@{}];
        if (![handler performRequests:@[request] error:&error]) {
            fprintf(stderr, "%s\n", error.localizedDescription.UTF8String);
            return 5;
        }

        NSArray<VNRecognizedTextObservation *> *observations = [request.results sortedArrayUsingComparator:^NSComparisonResult(VNRecognizedTextObservation *a, VNRecognizedTextObservation *b) {
            if (fabs(CGRectGetMidY(a.boundingBox) - CGRectGetMidY(b.boundingBox)) > 0.025) {
                return CGRectGetMidY(a.boundingBox) > CGRectGetMidY(b.boundingBox) ? NSOrderedAscending : NSOrderedDescending;
            }
            return CGRectGetMinX(a.boundingBox) < CGRectGetMinX(b.boundingBox) ? NSOrderedAscending : NSOrderedDescending;
        }];

        NSMutableArray *words = [NSMutableArray array];
        NSRegularExpression *regex = [NSRegularExpression regularExpressionWithPattern:@"\\S+" options:0 error:nil];
        __block NSInteger wordIndex = 0;

        for (VNRecognizedTextObservation *observation in observations) {
            VNRecognizedText *candidate = [[observation topCandidates:1] firstObject];
            if (!candidate) continue;
            NSString *text = candidate.string;
            NSArray<NSTextCheckingResult *> *matches = [regex matchesInString:text options:0 range:NSMakeRange(0, text.length)];

            for (NSTextCheckingResult *match in matches) {
                NSError *boxError = nil;
                VNRectangleObservation *box = [candidate boundingBoxForRange:match.range error:&boxError];
                if (!box || boxError) continue;
                CGRect rect = box.boundingBox;
                [words addObject:@{
                    @"id": [NSString stringWithFormat:@"w%ld", (long)wordIndex++],
                    @"text": [text substringWithRange:match.range],
                    @"x": @(CGRectGetMinX(rect) * 1000.0),
                    @"y": @((1.0 - CGRectGetMaxY(rect)) * 1000.0),
                    @"width": @(CGRectGetWidth(rect) * 1000.0),
                    @"height": @(CGRectGetHeight(rect) * 1000.0)
                }];
            }
        }

        NSData *json = [NSJSONSerialization dataWithJSONObject:words options:0 error:&error];
        if (!json) return 6;
        [[NSFileHandle fileHandleWithStandardOutput] writeData:json];
    }
    return 0;
}

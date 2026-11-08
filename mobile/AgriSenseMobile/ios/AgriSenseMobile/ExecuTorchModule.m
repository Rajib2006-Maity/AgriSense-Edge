// RN requires an Objective-C bridging file to expose @objc Swift methods
// to the JS bridge, even though the implementation lives in ExecuTorchModule.swift.
#import <React/RCTBridgeModule.h>

@interface RCT_EXTERN_MODULE(ExecuTorchModule, NSObject)

RCT_EXTERN_METHOD(preloadModels:(BOOL)useQuantized
                  resolver:(RCTPromiseResolveBlock)resolve
                  rejecter:(RCTPromiseRejectBlock)reject)

RCT_EXTERN_METHOD(classifyLeaf:(NSString *)imageBase64
                  useQuantized:(BOOL)useQuantized
                  resolver:(RCTPromiseResolveBlock)resolve
                  rejecter:(RCTPromiseRejectBlock)reject)

RCT_EXTERN_METHOD(segmentLesion:(NSString *)imageBase64
                  useQuantized:(BOOL)useQuantized
                  resolver:(RCTPromiseResolveBlock)resolve
                  rejecter:(RCTPromiseRejectBlock)reject)

RCT_EXTERN_METHOD(unloadModels:(RCTPromiseResolveBlock)resolve
                  rejecter:(RCTPromiseRejectBlock)reject)

@end

if __name__ == "__main__":
    import torch
    import torchvision

    torch.hub._validate_not_a_forked_repo = lambda a, b, c: True

    resnet50_model = torch.hub.load(
        "pytorch/vision:v0.10.0", "resnet50", pretrained=True
    )
    resnet50_model.eval()
    print(resnet50_model)

    example = torch.rand(1, 3, 244, 244)

    # Use torch.jit.trace to generate a torch.jit.ScriptModule via tracing.
    traced_script_module = torch.jit.trace(resnet50_model, example)

    traced_script_module.save(
        "/efs/model_inference/testResNetTensorRT/full_model_fast_V2.pt"
    )

    print("ResNet Saved")

    import numpy as np
    import time
    import torch.backends.cudnn as cudnn

    cudnn.benchmark = True

    def benchmark(
        model, input_shape=(1024, 1, 224, 224), dtype="fp32", nwarmup=50, nruns=10000
    ):
        input_data = torch.randn(input_shape)
        input_data = input_data.to("cuda")
        if dtype == "fp16":
            input_data = input_data.half()

        print("Warm up ...")
        with torch.no_grad():
            for _ in range(nwarmup):
                features = model(input_data)
        torch.cuda.synchronize()
        print("Start timing ...")
        timings = []
        with torch.no_grad():
            for i in range(1, nruns + 1):
                start_time = time.time()
                features = model(input_data)
                torch.cuda.synchronize()
                end_time = time.time()
                timings.append(end_time - start_time)
                if i % 10 == 0:
                    print(
                        "Iteration %d/%d, ave batch time %.2f ms"
                        % (i, nruns, np.mean(timings) * 1000)
                    )

        print("Input shape:", input_data.size())
        print("Output features size:", features.size())
        print("Average batch time: %.2f ms" % (np.mean(timings) * 1000))

    model = resnet50_model.eval().to("cuda")
    benchmark(model, input_shape=(1, 3, 224, 224), nruns=100)

    import torch_tensorrt

    # The compiled module will have precision as specified by "op_precision".
    # Here, it will have FP32 precision.
    trt_model_fp32 = torch_tensorrt.compile(
        model,
        inputs=[torch_tensorrt.Input((1, 3, 224, 224), dtype=torch.float32)],
        enabled_precisions=torch.float32,  # Run with FP32
        workspace_size=1 << 22,
    )

    benchmark(trt_model_fp32, input_shape=(1, 3, 224, 224), nruns=100)

    trt_model_fp16 = torch_tensorrt.compile(
        model,
        inputs=[torch_tensorrt.Input((1, 3, 224, 224), dtype=torch.half)],
        enabled_precisions={torch.half},  # Run with FP16
        workspace_size=1 << 22,
    )

    benchmark(trt_model_fp16, input_shape=(1, 3, 224, 224), dtype="fp16", nruns=100)

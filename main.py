import torch
from MNISTCnns import Pictures
import warnings
import torch.nn as nn
from MNISTCnns import LeNet5
from MNISTCnns import AlexNet
from MNISTCnns import testNet
from MNISTCnns import VGG16
from MNISTCnns import GoogleNetV1
from MNISTCnns import GoogleNetV2
from MNISTCnns import GoogleNetV3
from MNISTCnns import GoogleNetV4
from MNISTCnns import ResNet
from MNISTCnns import ResNeXt

# 忽略 NCCL 相关的 UserWarning
warnings.filterwarnings("ignore", category=UserWarning, message=".*NCCL.*")

"""
    测试部分CNN，使用MNIST和FashionMNIST数据集
"""


# 测试LeNet5
def test_LeNet5(_learning_rate=0.01, batch_size=256, scheduler_type='Origin'):
    model_LeNet = LeNet5.Lenet5().to('cuda:0')
    print(model_LeNet.__class__.__name__, '训练结果：')
    model_LeNet = nn.DataParallel(model_LeNet)
    train_data_le, test_data_le, train_loader_le, test_loader_le = LeNet5.Get_dataset()
    model_LeNet, loss_LeNet = testNet.test(model_LeNet, train_loader_le, test_loader_le,
                                           _learning_rate=_learning_rate,
                                           batch_size=batch_size,
                                           scheduler_type=scheduler_type)
    return model_LeNet, loss_LeNet


# 测试AlexNet
def test_AlexNet(_learning_rate=0.01, batch_size=128, scheduler_type='Origin'):
    model_AlexNet = AlexNet.AlexNet().to('cuda:0')
    print(model_AlexNet.__class__.__name__, '训练结果：')
    model_AlexNet= nn.DataParallel(model_AlexNet)
    train_data_al, test_data_al, train_loader_al, test_loader_al = AlexNet.Get_dataset()
    model_AlexNet, loss_AlexNet = testNet.test(model_AlexNet, train_loader_al, test_loader_al,
                                               _learning_rate=_learning_rate,
                                               batch_size=batch_size,
                                               scheduler_type=scheduler_type)
    return model_AlexNet, loss_AlexNet


# 测试Vgg16
def test_vgg16(_learning_rate=0.01, batch_size=64, scheduler_type='Origin'):
    model_vgg16 = VGG16.vgg_16().to('cuda:0')
    print(model_vgg16.__class__.__name__, '训练结果：')
    model_vgg16 = nn.DataParallel(model_vgg16)
    train_data_vgg, test_data_vgg, train_loader_vgg, test_loader_vgg = VGG16.Get_dataset()
    model_vgg16, loss_vgg = testNet.test(model_vgg16, train_loader_vgg, test_loader_vgg,
                                         _learning_rate=_learning_rate,
                                         batch_size=batch_size,
                                         scheduler_type=scheduler_type)
    return model_vgg16, loss_vgg


# 测试GoogleNet_v1
def test_GoogleNet_v1(_learning_rate=0.001, batch_size=64, scheduler_type='Origin'):
    model_GooV1 = GoogleNetV1.GoogleNet_V1().to('cuda:0')
    print(model_GooV1.__class__.__name__, '训练结果：')
    model_GooV1 = nn.DataParallel(model_GooV1)
    train_data_goo, test_data_goo, train_loader_goo, test_loader_goo = GoogleNetV1.Get_dataset()
    model_GooV1, loss_Goo1 = testNet.test(model_GooV1, train_loader_goo, test_loader_goo,
                                          _learning_rate=_learning_rate,
                                          batch_size=batch_size,
                                          scheduler_type=scheduler_type)
    return model_GooV1, loss_Goo1


# 测试GoogleNet_v2
def test_GoogleNet_v2(_learning_rate=0.01, batch_size=64, scheduler_type='Origin'):
    model_GooV2 = GoogleNetV2.GoogleNet_V2().to('cuda:0')
    print(model_GooV2.__class__.__name__, '训练结果：')
    model_GooV2= nn.DataParallel(model_GooV2)
    train_data_goo, test_data_goo, train_loader_goo, test_loader_goo = GoogleNetV2.Get_dataset()
    model_GooV2, loss_Goo2 = testNet.test(model_GooV2, train_loader_goo, test_loader_goo,
                                          _learning_rate=_learning_rate,
                                          batch_size=batch_size,
                                          scheduler_type=scheduler_type)
    return model_GooV2, loss_Goo2


# 测试GoogleNet_v3
def test_GoogleNet_v3(_learning_rate=0.01, batch_size=64, scheduler_type='Origin'):
    model_GooV3 = GoogleNetV3.GoogleNet_V3().to('cuda:0')
    print(model_GooV3.__class__.__name__, '训练结果：')
    model_GooV3= nn.DataParallel(model_GooV3)
    train_data_goo, test_data_goo, train_loader_goo, test_loader_goo = GoogleNetV3.Get_dataset()
    model_GooV3, loss_Goo3 = testNet.test(model_GooV3, train_loader_goo, test_loader_goo,
                                          _learning_rate=_learning_rate,
                                          batch_size=batch_size,
                                          scheduler_type=scheduler_type)
    return model_GooV3, loss_Goo3


# 测试GoogleNet_v4
def test_GoogleNet_v4(_learning_rate=0.001, batch_size=64, scheduler_type='Origin'):
    model_GooV4 = GoogleNetV4.GoogleNet_V4().to('cuda:0')
    print(model_GooV4.__class__.__name__, '训练结果：')
    model_GooV4 = nn.DataParallel(model_GooV4)
    train_data_goo, test_data_goo, train_loader_goo, test_loader_goo = GoogleNetV4.Get_dataset()
    model_GooV4, loss_Goo4 = testNet.test(model_GooV4, train_loader_goo, test_loader_goo,
                                          _learning_rate=_learning_rate,
                                          batch_size=batch_size,
                                          scheduler_type=scheduler_type)
    return model_GooV4, loss_Goo4


# 测试ResNet
def test_ResNet(_learning_rate=0.01, batch_size=128, scheduler_type='Origin'):
    model_Res = ResNet.ResNet().to('cuda:0')
    print(model_Res.__class__.__name__, '训练结果：')
    model_Res = nn.DataParallel(model_Res)
    train_data_res, test_data_res, train_loader_res, test_loader_res = ResNet.Get_dataset()
    model_Res, loss_Res = testNet.test(model_Res, train_loader_res, test_loader_res,
                                       _learning_rate=_learning_rate,
                                       batch_size=batch_size,
                                       scheduler_type=scheduler_type)
    return model_Res, loss_Res


# 测试ResNeXt
def test_ResNeXt(_learning_rate=0.01, batch_size=64, scheduler_type='Origin'):
    model_ResNeXt = ResNeXt.ResNeXt().to('cuda:0')
    print(model_ResNeXt.__class__.__name__, '训练结果：')
    model_ResNeXt = nn.DataParallel(model_ResNeXt)
    train_data_resX, test_data_resX, train_loader_resX, test_loader_resX = ResNeXt.Get_dataset()
    model_ResNeXt, loss_ResNeXt = testNet.test(model_ResNeXt, train_loader_resX, test_loader_resX,
                                               _learning_rate=_learning_rate,
                                               batch_size=64,
                                               scheduler_type=scheduler_type)
    return model_ResNeXt, loss_ResNeXt


# GoogleNet_v1_1, loss_v1_1 = test_GoogleNet_v1()
Google_v1_2, loss_v1_2 = test_GoogleNet_v1(scheduler_type='Factor')
# torch.save(Google_v1, 'GoogleV1.pth')
#
# Google_v2, loss_v2 = test_GoogleNet_v2()
# torch.save(Google_v2, 'GoogleV2.pth')

# Google_v3, loss_v3 = test_GoogleNet_v3()
# torch.save(Google_v3, 'GoogleV3.pth')
# Pictures.save_picture(range(len(loss_v3)), loss_v3, 'loss_GoogleNetV3')


# Google_v4, loss_v4 = test_GoogleNet_v4()
# torch.save(Google_v4, 'GoogleV4.pth')
# Pictures.save_picture(range(len(loss_v4)), loss_v4, 'loss_GoogleNetV4')

# res, loss_res = test_ResNet()  #
# torch.save(res, 'ResNet.pth')
# Pictures.save_picture(range(len(loss_res)), loss_res, 'loss_ResNet')

# vgg, loss_vgg_ = test_vgg16()
# torch.save(vgg, 'vgg16.pth')
# Pictures.save_picture(range(len(loss_vgg_)), loss_vgg_, 'loss_vgg16')

# resX, loss_resX = test_ResNeXt()
# torch.save(resX, 'resNeXt.pth')
# Pictures.save_picture(range(len(loss_resX)), loss_resX, 'loss_resNeXt')

# x = torch.rand((1, 1, 299, 299))
# for layer in GoogleNetV4.GoogleNet_V4().net:
#     x = layer(x)
#     print(layer.__class__.__name__, 'output:\t', x.shape)

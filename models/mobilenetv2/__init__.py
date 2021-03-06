from .mobilenetv2 import MobileNetV2, MobileNetV2VGGBlock, MobileNetV2VGGBlockTemper1
from .mobile_net_temper_wrapper import MobileNetTemperWrapper

configs = []

orig_vggblock = {
    'orig': MobileNetV2,
    'tempered': MobileNetV2VGGBlock,
    'orig_module_names' : [
        'orig.feature_1.0',
        'orig.feature_1.1',
        'orig.feature_1.2',
        'orig.feature_1.3',
        'orig.feature_2.0',
        'orig.feature_2.1',
        'orig.feature_2.2',
        'orig.feature_4.0',
        'orig.feature_4.1',
        'orig.feature_4.2',
        'orig.feature_4.3',
        'orig.feature_4.4',
        'orig.feature_4.5',
        'orig.feature_4.6',
        'orig.feature_6.0',
        'orig.feature_6.1',
        'orig.feature_6.2',
        'orig.feature_6.3',
    ],
    "tempered_module_names" : [
        'tempered.feature_1.0',
        'tempered.feature_1.1',
        'tempered.feature_1.2',
        'tempered.feature_1.3',
        'tempered.feature_2.0',
        'tempered.feature_2.1',
        'tempered.feature_2.2',
        'tempered.feature_4.0',
        'tempered.feature_4.1',
        'tempered.feature_4.2',
        'tempered.feature_4.3',
        'tempered.feature_4.4',
        'tempered.feature_4.5',
        'tempered.feature_4.6',
        'tempered.feature_6.0',
        'tempered.feature_6.1',
        'tempered.feature_6.2',
        'tempered.feature_6.3',
    ],
    'is_trains' : [
        True,
        True,
        True,
        True,
        True,
        True,
        True,
        True,
        True,
        True,
        True,
        True,
        True,
        True,
        True,
        True,
        True,
        True,
    ]
}
configs.append(orig_vggblock)

vgg_vggblocktemper1 = {
    'orig': MobileNetV2VGGBlock,
    'tempered': MobileNetV2VGGBlockTemper1,
    'orig_module_names' : [
        'orig.feature_1.0',
        'orig.feature_1.1',
        'orig.feature_1.2',
        'orig.feature_1.3',
        'orig.feature_2.0',
        'orig.feature_2.1',
        'orig.feature_2.2',
        'orig.feature_4.0',
        'orig.feature_4.1',
        ['orig.feature_4.2', 'orig.feature_4.3'],
        'orig.feature_4.4',
        ['orig.feature_4.5', 'orig.feature_4.6'],
        'orig.feature_6.0',
        ['orig.feature_6.1', 'orig.feature_6.2'],
        'orig.feature_6.3',
    ],
    "tempered_module_names" : [
        'tempered.feature_1.0',
        'tempered.feature_1.1',
        'tempered.feature_1.2',
        'tempered.feature_1.3',
        'tempered.feature_2.0',
        'tempered.feature_2.1',
        'tempered.feature_2.2',
        'tempered.feature_4.0',
        'tempered.feature_4.1',
        'tempered.feature_4.2',
        'tempered.feature_4.3',
        'tempered.feature_4.4',
        'tempered.feature_6.0',
        'tempered.feature_6.1',
        'tempered.feature_6.2',
    ],
    "is_trains" : [
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        True,
        False,
        True,
        False,
        True,
        False,
    ]
}
# dataset settings
dataset_type = 'CocoDataset'

############ on pub dataset ##############
data_root = '../data/clipart1k'

# #UODD
# metainfo = dict(
# classes = (
#     "seacucumber",
#     "seaurchin",
#     "scallop",
# )
# )

#DIOR
# metainfo = dict(
# classes = (
#    "Expressway-Service-area",
#    "Expressway-toll-station",
#    "airplane",
#    "airport",
#    "baseballfield",
#    "basketballcourt",
#    "bridge",
#    "chimney",
#    "dam",
#    "golffield",
#    "groundtrackfield",
#    "harbor",
#    "overpass",
#    "ship",
#    "stadium",
#    "storagetank",
#    "tenniscourt",
#    "trainstation",
#    "vehicle",
#    "windmill",
# )
# )

# # east/south
# metainfo = dict(
# classes = (
#     "Anticyclone",
#     "Cyclone",
# )
# )

#ArTaxOr
# metainfo = dict(
# classes = (
#     "Araneae",
#     "Coleoptera",
#     "Diptera",
#     "Hemiptera",
#     "Hymenoptera",
#     "Lepidoptera",
#     "Odonata",
# )
# )

#clipart1k
metainfo = dict(
 classes = (
     "sheep",
     "chair",
     "boat",
     "bottle",
     "diningtable",
     "sofa",
     "cow",
     "motorbike",
     "car",
     "aeroplane",
     "cat",
     "train",
     "person",
     "bicycle",
     "pottedplant",
     "bird",
     "dog",
     "bus",
     "tvmonitor",
     "horse"
 )
 )



#FISH
# metainfo = dict(
# classes = (
#     "fish",
# )
# )

#NEUDET
# metainfo = dict(
# classes = (
#     "crazing",
#     "inclusion",
#     "patches",
#     "pitted_surface",
#     "rolled-in_scale",
#     "scratches",
# )
# )


#train_ann_file = 'annotations/1_shot.json'
train_ann_file = 'annotations/5_shot.json'
#train_ann_file = 'annotations/10_shot.json'



############################
backend_args = None

train_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='RandomChoice',
        transforms=[
            [
                dict(
                    type='RandomChoiceResize',
                    scales=[(480, 1333), (512, 1333), (544, 1333), (576, 1333),
                            (608, 1333), (640, 1333), (672, 1333), (704, 1333),
                            (736, 1333), (768, 1333), (800, 1333)],
                    keep_ratio=True)
            ],
            [
                dict(
                    type='RandomChoiceResize',
                    # The radio of all image in train dataset < 7
                    # follow the original implement
                    scales=[(400, 4200), (500, 4200), (600, 4200)],
                    keep_ratio=True),
                dict(
                    type='RandomCrop',
                    crop_type='absolute_range',
                    crop_size=(384, 600),
                    allow_negative_crop=True),
                dict(
                    type='RandomChoiceResize',
                    scales=[(480, 1333), (512, 1333), (544, 1333), (576, 1333),
                            (608, 1333), (640, 1333), (672, 1333), (704, 1333),
                            (736, 1333), (768, 1333), (800, 1333)],
                    keep_ratio=True)
            ]
        ]),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor', 'flip', 'flip_direction', 'text',
                   'custom_entities'))
]

test_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='FixScaleResize', scale=(800, 1333), keep_ratio=True),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor', 'text', 'custom_entities'))
]

train_real_dataset=dict(
    type=dataset_type,
    data_root=data_root,
    ann_file=train_ann_file,
    metainfo=metainfo,
    data_prefix=dict(img='train/'),
    pipeline=train_pipeline,
    filter_cfg=dict(filter_empty_gt=False),
    return_classes=True)

train_dataloader = dict(
    batch_size=4,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    batch_sampler=dict(type='AspectRatioBatchSampler'),
    dataset=train_real_dataset,
    )


val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        # ann_file='annotations/10_shot.json',
        ann_file=data_root + '/annotations/test.json',
        data_prefix=dict(img='test/'),
        test_mode=True,
        metainfo=metainfo,
        pipeline=test_pipeline,
        return_classes=True))

test_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=data_root + '/annotations/test.json',
        data_prefix=dict(img='test/'),
        test_mode=True,
        metainfo=metainfo,
        pipeline=test_pipeline,
        return_classes=True))

val_evaluator = dict(
    type='CocoMetric',
    # ann_file=data_root + 'annotations/10_shot.json',
    ann_file=data_root + '/annotations/test.json',
    metric='bbox',
    classwise=True,
    format_only=False,
    backend_args=backend_args)


test_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + '/annotations/test.json',
    metric='bbox',
    classwise=True,
    format_only=False,
    backend_args=backend_args)
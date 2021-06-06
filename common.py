from collections import OrderedDict
class Evaluation3D:
    CATEGORY_LIST = [
        "02691156_airplane",
        "02828884_bench",
        "02933112_cabinet",
        "02958343_car",
        "03001627_chair",
        "03211117_display",
        "03636649_lamp",
        "03691459_speaker",
        "04090263_rifle",
        "04256520_couch",
        "04379243_table",
        "04401088_phone",
        "04530566_vessel",
    ]
    CATEGORY_IDS = [name[:8] for name in CATEGORY_LIST]
    CATEGORY_NAMES = [name.split("_")[1] for name in CATEGORY_LIST]
    NUM_POINTS = 4096

    CATEGORY_COUNTS = [
        809,
        364,
        315,
        1500,
        1356,
        219,
        464,
        324,
        475,
        635,
        1702,
        211,
        388,
    ]
    object_to_reconstruct = OrderedDict(
    {
        "02691156": [
            "d18592d9615b01bbbc0909d98a1ff2b4",
            "d18f2aeae4146464bd46d022fd7d80aa",
            "d199612c22fe9313f4fb6842b3610149",
        ],
        "02828884": [
            "c8298f70c094a9fd25d3c528e036a532",
            "c83b3192c338527a2056b4bd5d870b47",
            "c8802eaffc7e595b2dc11eeca04f912e",
        ],
        "02933112": [
            "aff3488d05343a89e42b7a6468e7283f",
            "b06b351b939e279bc5ff6d1af2135fc9",
            "b0709afab8a3d9ce7e65d4ecde1c77ce",
        ],
        "02958343": [
            "cbc6e31c744ef872b34a6368f13f8b72",
            "cbc946b4f4c022305e524bb779a94481",
            "cbd0b4a0d264c856f35cb5c94d4343bf",
        ],
        "03001627": [
            "cbc47018135fc1b1462977c6d3c24550",
            "cbc5e6fce716e48ea28e529ba1f4836e",
            "cbc76d55a04d5b2e1d9a8cea064f5297",
        ],
        "03211117": [
            "d7ab9503d7f6dac6b4382097c3e8bcf7",
            "d7b87d0083bf5568fd28950562697757",
            "d8142f27c166cc21f103e4fb531505b4",
        ],
        "03636649": [
            "d13f1adad399c9f1ea93fe4e1ab627a2",
            "d153ae6d65b31e00fcb8d8c6d4df8143",
            "d16bb6b2f26084556acbef8d3bef8f28",
        ],
        "03691459": [
            "c90cbb0458648665da49a3feeb6532eb",
            "c91e878553979be9c5c378bd9e63485",
            "c91f926711d5e8261d485f425cc21556",
        ],
        "04090263": [
            "ca25a955adaa031812b38b1d99376c0b",
            "ca2bafb1ba4b97a1683e3750b53385d5",
            "ca4e431c75a8242445e0c3a4b827d51a",
        ],
        "04256520": [
            "cc644fad0b76a441d84c7dc40ac6d743",
            "cc7b690e4d86b471397aad305ec14786",
            "cc906e84c1a985fe80db6871fa4b6f35",
        ],
        "04379243": [
            "cd44665771f7b7d2b2000d40d3899456",
            "cd4e8748514e028642d23b95defe1ce5",
            "cd5f235344ff4c10d5b24cafb84903c7",
        ],
        "04401088": [
            "d2bce5fd1fd04beeb3db4664acec42ef",
            "d2f3eb92a31649647c17b7a9bb17a24",
            "d37afdca0c48251044b992023e0d3ef0",
        ],
        "04530566": [
            "cd65ea1bb0e091d5a1ea2dd0a4cf317e",
            "cd67f7d1ba943b162f84cb7932f866fd",
            "cdaff2fe98efb90058a8952c93ff9829",
        ],
    })

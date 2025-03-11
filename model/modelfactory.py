import numpy as np

class ModelFactory():
    def __init__(self):
        pass

    @staticmethod
    def get_model(model_type, dataset, input_dimension=6, output_dimension=6, width=300):

        if "Sin" == dataset:

            if model_type == "representation":

                hidden_size = width
                return [

                    {"name": 'linear', "adaptation": False, "meta": True,
                     "config": {"out": hidden_size, "in": input_dimension}},
                    {"name": 'relu'},
                    {"name": 'linear', "adaptation": False, "meta": True,
                     "config": {"out": hidden_size, "in": hidden_size}},
                    {"name": 'relu'},
                    {"name": 'linear', "adaptation": False, "meta": True,
                     "config": {"out": hidden_size, "in": hidden_size}},
                    {"name": 'relu'},
                    {"name": 'linear', "adaptation": False, "meta": True,
                     "config": {"out": hidden_size, "in": hidden_size}},
                    {"name": 'relu'},
                    {"name": 'linear', "adaptation": False, "meta": True,
                     "config": {"out": hidden_size, "in": hidden_size}},
                    {"name": 'relu'},
                    {"name": 'linear', "adaptation": True, "meta": True,
                     "config": {"out": output_dimension, "in": hidden_size}}
                ]

        # elif dataset == "omniglot":
        #     channels = 256
        #
        #     return [
        #         {"name": 'conv2d', "adaptation": False, "meta": True,
        #          "config": {"out-channels": channels, "in-channels": 1, "kernal": 3, "stride": 2, "padding": 0}},
        #         {"name": 'relu'},
        #
        #         {"name": 'conv2d', "adaptation": False, "meta": True,
        #          "config": {"out-channels": channels, "in-channels": channels, "kernal": 3, "stride": 1,
        #                     "padding": 0}},
        #         {"name": 'relu'},
        #
        #         {"name": 'conv2d', "adaptation": False, "meta": True,
        #          "config": {"out-channels": channels, "in-channels": channels, "kernal": 3, "stride": 2,
        #                     "padding": 0}},
        #         {"name": 'relu'},
        #         #
        #         {"name": 'conv2d', "adaptation": False, "meta": True,
        #          "config": {"out-channels": channels, "in-channels": channels, "kernal": 3, "stride": 1,
        #                     "padding": 0}},
        #         {"name": 'relu'},
        #
        #         {"name": 'conv2d', "adaptation": False, "meta": True,
        #          "config": {"out-channels": channels, "in-channels": channels, "kernal": 3, "stride": 2,
        #                     "padding": 0}},
        #         {"name": 'relu'},
        #
        #         {"name": 'conv2d', "adaptation": False, "meta": True,
        #          "config": {"out-channels": channels, "in-channels": channels, "kernal": 3, "stride": 2,
        #                     "padding": 0}},
        #         {"name": 'relu'},
        #
        #         {"name": 'flatten'},
        #         # {"name": 'rotate'},
        #         {"name": 'rep'},
        #
        #         {"name": 'linear', "adaptation": True, "meta": True,
        #          "config": {"out": 1000, "in": 9 * channels}}
        #
        #     ]

        elif dataset == "omniglot":
            channels = 256

            return [
                # RLN
                {"name": 'conv2d', "adaptation": False, "meta": True,
                 "config": {"out-channels": channels, "in-channels": 1, "kernal": 3, "stride": 2, "padding": 0}},
                {"name": 'relu'},

                {"name": 'conv2d', "adaptation": False, "meta": True,
                 "config": {"out-channels": channels, "in-channels": channels, "kernal": 3, "stride": 1,
                            "padding": 0}},
                {"name": 'relu'},

                {"name": 'conv2d', "adaptation": False, "meta": True,
                 "config": {"out-channels": channels, "in-channels": channels, "kernal": 3, "stride": 2,
                            "padding": 0}},
                {"name": 'relu'},
                #
                {"name": 'conv2d', "adaptation": False, "meta": True,
                 "config": {"out-channels": channels, "in-channels": channels, "kernal": 3, "stride": 1,
                            "padding": 0}},
                {"name": 'relu'},

                {"name": 'conv2d', "adaptation": False, "meta": True,
                 "config": {"out-channels": channels, "in-channels": channels, "kernal": 3, "stride": 2,
                            "padding": 0}},
                {"name": 'relu'},

                {"name": 'conv2d', "adaptation": False, "meta": True,
                 "config": {"out-channels": channels, "in-channels": channels, "kernal": 3, "stride": 2,
                            "padding": 0}},
                {"name": 'relu'},

                {"name": 'flatten'},
                # {"name": 'rotate'},


                {"name": 'linear', "adaptation": False, "meta": True,
                 "config": {"out": 1000, "in": 9 * channels}},

                {"name": 'rep'},

                # PLN
                {"name": 'relu'},

                {"name": 'linear', "adaptation": True, "meta": True,
                 "config": {"out": 1000, "in": 1000}}

            ]

        elif dataset == "mnist":

            channels = 64

            return [
                # 28
                {"name": 'conv2d', "adaptation": False, "meta": True,
                 "config": {"out-channels": channels, "in-channels": 1, "kernal": 3, "stride": 1, "padding": 1}},
                # 28
                {"name": 'relu'},

                {"name": 'conv2d', "adaptation": False, "meta": True,
                 "config": {"out-channels": channels, "in-channels": channels, "kernal": 3, "stride": 1, "padding": 1}},

                {"name": 'relu'},

                {"name": 'conv2d', "adaptation": False, "meta": True,
                 "config": {"out-channels": channels * 2, "in-channels": channels, "kernal": 3, "stride": 3,
                            "padding": 1}},
                # 10
                {"name": 'relu'},

                {"name": 'conv2d', "adaptation": False, "meta": True,
                 "config": {"out-channels": channels * 2, "in-channels": channels * 2, "kernal": 3, "stride": 1,
                            "padding": 1}},

                {"name": 'relu'},

                {"name": 'flatten'},


                {"name": 'linear', "adaptation": False, "meta": True,
                 "config": {"out": 10, "in": 10 * 10 * channels * 2}},

                {"name": 'rep'},

                # PLN
                {"name": 'relu'},

                {"name": 'linear', "adaptation": True, "meta": True,
                 "config": {"out": 10, "in": 10}}

            ]


        else:
            print("Unsupported model; either implement the model in model/ModelFactory or choose a different model")
            assert (False)

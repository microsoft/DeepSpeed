from .transformer_inference import InferenceBuilder

unittest_file_dict = {
    "Activation": [
        'csrc/transformer/inference/csrc/gelu.cu',
        'csrc/transformer/inference/csrc/activation_binding.cpp'
    ]
}
unittest_macro_dict = {"Activation": ['-DACTIVATION_UNITTEST']}


class InferenceUnitTestBuilder(InferenceBuilder):
    NAME = "deepspeed_unittest"

    def __init__(self, test_name=None):
        self.test_name = test_name
        super().__init__(name=self.NAME)

    def builder_macros(self):
        if self.test_name in unittest_macro_dict.keys():
            return unittest_macro_dict[self.test_name]
        else:
            raise RuntimeError("Unittest macro not provided.")

    def sources(self):
        if self.test_name is None:
            return unittest_file_dict.values()

        if self.test_name in unittest_file_dict.keys():
            return unittest_file_dict[self.test_name]
        else:
            raise RuntimeError("Unittest source files not provided.")

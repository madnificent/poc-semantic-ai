class Preprocessor:
    def preprocess(self, value):
        return value

    def __call__(self, *args, **kwargs):
        return self.preprocess(*args, **kwargs)

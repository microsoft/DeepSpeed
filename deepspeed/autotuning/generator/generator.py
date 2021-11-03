import itertools


class GridSearchGenerator():
    def __init__(self, sample_domain, sample_size):
        self.sample_domain = sample_domain
        self.sample_size = sample_size

    def __iter__(self):
        domain_size = len(self.sample_domain)
        for idx in range(0, domain_size, self.sample_size):
            end = min(idx + self.sample_size, domain_size)
            yield self.sample_domain[idx:end]


class RandomGenerator():
    def __init__(self, sample_domain, sample_size):
        self.sample_domain = set(sample_domain)
        self.sample_size = sample_size

    def __iter__(self):
        import random
        while True:
            domain_size = min(self.sample_size, len(self.sample_domain))
            if domain_size == 0:
                break
            sample = random.sample(self.sample_domain, domain_size)
            self.sample_domain = [x for x in self.sample_domain if x not in sample]
            yield sample


class FindTuneSpace:
    def __init__(self, generator):
        self.generator = generator

    def find_best(self):
        best = None
        best_score = 0
        for space in self.generator:
            print(space)
            score = self.evaluate(space)
            if best == None or score < best_score:
                best = space
                best_score = score
        return best

    def evaluate(self, space):
        return sum(space)


def run():
    params = range(1, 20)
    space = FindTuneSpace(GridSearchGenerator(params, 2))
    print(f'best is {space.find_best()}')


def main():
    run()


if __name__ == "__main__":
    main()

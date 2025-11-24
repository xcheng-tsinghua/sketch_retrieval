from data import retrieval_datasets






if __name__ == '__main__':
    aset = retrieval_datasets.QMULDataset()
    print(len(aset))
    print(aset[0][0].size(), aset[0][1].size())






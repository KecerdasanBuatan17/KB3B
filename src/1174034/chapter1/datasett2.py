from sklearn import datasets #mengimport class dataset dari scikit learn library 
iris = datasets.load_iris() # memuat dan memasukkan dataset iris ke variabel bernama iris 
digits = datasets.load_digits() #memuat dan memasukkan dataset digits ke variabel digits

print(digits.data) #memberikan akses ke fitur yang dapat digunakan untuk mengklasifikasikan sampel digit dan menampilkannya di console

digits.target #memberikan informasi tentang data yang berhubungan atau juga dapat dijadikan sebagai label

digits.images[0] #Data selalu berupa array 2D, shape (n_samples, n_features), meskipun data aslinya mungkin memiliki bentuk yang berbeda.

from sklearn import datasets 
#pada baris ini merupakan sebuah perintah untuk mengimport class datasets dari packaged sklearn
iris = datasets.load_iris() 
#pada baris kedua ini dimana iris merupakan suatu estimator/parameter yang berfungsi untuk mengambil data pada item datasets.load iris
digits = datasets.load_digits() 
#pada baris ketiga ini dimana digits merupakan suatu estimator/parameter yang berfungsi untuk mengambil data pada item datasets.load digits
print(digits.data) 
#pada baris keempat ini merupakan perintah yang berfungsi untuk menampilkan estimator/parameter yang dipanggil pada item digits.data dan menampilkan outputannya
digits.target 
#barisan ini untuk mengambil target pada estimator/parameter digits dan menampilkan outputannya
digits.images[0] 
#barisan ini untuk mengambil images[0] pada estimator/parameter digits dan menampilkan outputannyal

E-Ticaret Lojistiğinde Teslimat Süresi Tahmini
================
Furkan Şahin
2026-02-25

``` r
summary(model_data)
```

    ##  delivery_duration freight_value    product_weight_g product_volume  
    ##  Min.   :  0.00    Min.   :  0.00   Min.   :    0    Min.   :   168  
    ##  1st Qu.:  6.00    1st Qu.: 13.08   1st Qu.:  300    1st Qu.:  2856  
    ##  Median : 10.00    Median : 16.28   Median :  700    Median :  6512  
    ##  Mean   : 11.97    Mean   : 19.97   Mean   : 2092    Mean   : 15191  
    ##  3rd Qu.: 15.00    3rd Qu.: 21.15   3rd Qu.: 1800    3rd Qu.: 18240  
    ##  Max.   :208.00    Max.   :409.68   Max.   :40425    Max.   :296208  
    ##  NA's   :8                          NA's   :1        NA's   :1       
    ##  is_differente_state is_differente_city is_differente_zip
    ##  Min.   :0.000       Min.   :0.000      Min.   :0.0000   
    ##  1st Qu.:0.000       1st Qu.:1.000      1st Qu.:1.0000   
    ##  Median :1.000       Median :1.000      Median :1.0000   
    ##  Mean   :0.638       Mean   :0.948      Mean   :0.9998   
    ##  3rd Qu.:1.000       3rd Qu.:1.000      3rd Qu.:1.0000   
    ##  Max.   :1.000       Max.   :1.000      Max.   :1.0000   
    ## 

Bir önceki çalışmamda SQL ile veriyi hazırlarken sadece ürünün teslim
edildiği satırları seçmiştim ama ‘delivery_duration’ sütunu yani ürünün
kaç günde teslim edildiğini veren sütunda 0 olan gözlemler görüyoruz. Bu
aynı günde teslim edilen ürünlerden kaynaklanıyor.

Ürünlerin ağırlığı sütununu incelediğimizde 0 olan gözlemler görüyoruz
bunun sebebi bu gözlemler dijital ürünler olduğu içindir. Bu ürünler
kargo gerektirmediği için modelimizi yanıltır bu yüzden veriden
çıkaracağım.

0 ve 1’lerden oluşan sütunlar kategoriktir. Bu sütunları faktör hale
getireceğim.

Son olarak bazı değişkenlerde kayıp gözlemler görüyoruz. Modeli kurmadan
önce kayıp gözlemleri dolduracağım.

``` r
model_data <- model_data %>%
  filter(product_weight_g != 0)

model_data$is_differente_city <- 
factor(model_data$is_differente_city)

model_data$is_differente_state <- 
factor(model_data$is_differente_state)

model_data$is_differente_zip <- 
factor(model_data$is_differente_zip)

imputed <- mice(model_data, printFlag = FALSE)
imputed_data <- complete(imputed)
```

![](estimate_delivery_time_files/figure-gfm/unnamed-chunk-3-1.png)<!-- -->

Veride negatif değerler bulunmuyor bu yüzden sağa çarpıklık
gözlemleniyor. Aynı zamanda bağımlı değişkenim kesiklidir, bu yüzden
modelimi negatif binom, yada poisson modelleri üzerinden kuracağım.

![](estimate_delivery_time_files/figure-gfm/unnamed-chunk-4-1.png)<!-- -->

Bağımlı değişken ile bağımsız değişkenler arasında düşük korelasyon
gözlemleniyor. Diğer yandan bağımsız değişkenler arasında yüksek
korelasyon gözlemleniyor. Çoklu Doğrusal Bağlantı sorunu ortaya
çıkabilir.

Eğitim test parçalanması

``` r
set.seed(123)
index <- sample(1:nrow(imputed_data), 
0.8 * nrow(imputed_data))

train <- imputed_data[index, ]
test <- imputed_data[-index, ]
```

Poisson ve negatif binom modelleri kuruyorum.

``` r
model_poisson <- step(glm(delivery_duration ~ ., 
family = poisson(link = "log"), 
data = train), 
direction = 'both',
trace = FALSE)

model_nb <- step(glm.nb(delivery_duration ~ ., 
data = train),
direction = 'both',
trace = FALSE)
```

Poisson modeli için aşırı yayılım kontrolü

``` r
dispersion_oran <- summary(model_poisson)$deviance / 
summary(model_poisson)$df.residual

cat("Aşırı Yayılım Oranı:", dispersion_oran, "\n")
```

    ## Aşırı Yayılım Oranı: 4.53097

Negatif binom modeli için aşırı yayılım kontrolü

``` r
dispersion_oran2 <- summary(model_nb)$deviance / 
summary(model_nb)$df.residual

cat("Aşırı Yayılım Oranı:", dispersion_oran2, "\n")
```

    ## Aşırı Yayılım Oranı: 1.0116

AIC değerlerini karşılaştırma

``` r
AIC(model_poisson, model_nb)
```

    ##               df      AIC
    ## model_poisson  7 748530.9
    ## model_nb       7 560601.9

Test seti üzerinde tahmin yapma

``` r
pred_poisson <- predict(model_poisson, 
newdata = test, type = "response")

pred_nb <- predict(model_nb, 
newdata = test, type = "response")
```

Gerçek değerler

``` r
gercek <- test$delivery_duration
```

RMSE ve MAE hesaplama (Negatif Binom vs Poisson)

``` r
rmse_p <- sqrt(mean((gercek - pred_poisson)^2, 
na.rm=TRUE))

rmse_nb <- sqrt(mean((gercek - pred_nb)^2, 
na.rm=TRUE))

mae_p <- mean(abs(gercek - pred_poisson), na.rm=TRUE)
mae_nb <- mean(abs(gercek - pred_nb), na.rm=TRUE)

cat("Poisson Model -> RMSE:", rmse_p, " MAE:", mae_p, "\n")
```

    ## Poisson Model -> RMSE: 8.388187  MAE: 5.528223

``` r
cat("Negatif Binom -> RMSE:", rmse_nb, " MAE:", mae_nb, "\n")
```

    ## Negatif Binom -> RMSE: 8.410546  MAE: 5.533909

İlk olarak sayım verisine uygun olan Poisson modelini kurdum. Ancak
varyans testini yaptığımda kalıntı sapması oranının 4.53 çıktığını
gördüm. Teslimat sürelerindeki değişkenlik, modelin varsaydığından
yaklaşık 4.5 kat daha fazla. Yani verimizde şiddetli bir aşırı yayılım
(overdispersion) var ve Poisson modeli bu veriyi açıklamakta yetersiz
kalıyor.

Poisson’un bu zafiyetini gidermek için aşırı yayılımı tolere edebilen
Negatif Binom modeline geçiş yaptım. Yayılım testini tekrarladığımda
oranın 1.01’e, yani kabul edilen 1 seviyesine düştüğünü gözlemledim.

Poisson ve negatif binom modellerinin AIC, RMSE ve MAE test sonuçlarını
karşılaştırdığımda da negatif binom modelinin daha başarılı olduğunu
gözlemledim.

``` r
options(width = 45)
vif(model_nb)
```

    ##       freight_value    product_weight_g 
    ##            1.989953            3.193820 
    ##      product_volume is_differente_state 
    ##            3.037891            1.263975 
    ##  is_differente_city 
    ##            1.100095

Çoklu doğrusal bağlantı problemi gözlenmiyor.

Katsayıların yorumlanması

``` r
options(width = 45)
exp(coef(model_nb))
```

    ##          (Intercept)        freight_value 
    ##             5.103218             1.003827 
    ##     product_weight_g       product_volume 
    ##             1.000004             1.000001 
    ## is_differente_state1  is_differente_city1 
    ##             1.798457             1.415993

Herhangi bir koşul altında olmadan ortalama teslimat süresi 5 gündür.
Kargo ücreti, ağırlığı ve hacminin teslimat süresine katkısı yok denecek
kadar azdır. Satıcı ve müşterilerin farklı bölgedelerde bulunması
teslimat süresini %79 ve başka şehirlerde olmaları %41 arttırmaktadır.

p-value

``` r
lrtest(model_nb)
```

    ## Likelihood ratio test
    ## 
    ## Model 1: delivery_duration ~ freight_value + product_weight_g + product_volume + 
    ##     is_differente_state + is_differente_city
    ## Model 2: delivery_duration ~ 1
    ##   #Df  LogLik Df Chisq Pr(>Chisq)    
    ## 1   7 -280294                        
    ## 2   2 -291483 -5 22378  < 2.2e-16 ***
    ## ---
    ## Signif. codes:  
    ##   0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1  ' ' 1

Negatif Binom modelin için McFadden Pseudo R-kare hesaplaması

``` r
pseudo_r2 <- 1 - (summary(model_nb)$deviance / 
summary(model_nb)$null.deviance)

cat("McFadden Pseudo R2:", pseudo_r2)
```

    ## McFadden Pseudo R2: 0.2231632

Model anlamlı ve açıklayıcıdır.


s displayed.

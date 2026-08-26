# Mektup Düzeltici

El yazısıyla yazılmış İngilizce bir mektubun fotoğrafını analiz eder; yanlış kelime veya ifadelerin üzerini çizip doğru hâllerini fotoğrafın üstüne yazar. Sonuç PNG olarak indirilebilir.

## Kurulum

Node.js 20 veya daha yenisi ve yerel LanguageTool gerekir. Google Cloud Vision tarafından okunan aynı metin OpenAI GPT ve Google Gemini ile ayrı ayrı düzeltilip yan yana gösterilir. LanguageTool kurulumu:

```bash
brew install languagetool
```

Bağımlılıkları kurmak için proje klasöründe bir kez `npm install` çalıştırın.

1. `.env.example` dosyasını `.env` adıyla kopyalayın.
2. `.env` içindeki `OPENAI_API_KEY` değerini kendi OpenAI API anahtarınızla doldurun.
3. Google AI Studio'dan oluşturduğunuz anahtarı `GEMINI_API_KEY` alanına yazın.
4. Google Cloud projesinde Cloud Vision API'yi etkinleştirin. Servis hesabı JSON dosyanızın mutlak yolunu `GOOGLE_APPLICATION_CREDENTIALS` alanına yazın. Alternatif olarak `GOOGLE_CLOUD_VISION_API_KEY` alanına bir `AIza...` API anahtarı yazabilirsiniz.
5. Uygulamayı başlatın:

```bash
npm start
```

6. Tarayıcıda `http://localhost:3000` adresini açın.

## Nasıl çalışır?

- Fotoğraf tarayıcıdan sunucuya base64 veri olarak gönderilir.
- Google Cloud Vision `DOCUMENT_TEXT_DETECTION` görünür kelimeleri ve bounding-box koordinatlarını bir kez üretir.
- Aynı OCR metni ve aynı fotoğraf OpenAI GPT ile Google Gemini'ye paralel gönderilir.
- İki yapay zekâ sonucu aynı dilbilgisi, diff, OpenCV temizleme ve Canvas çizim adımlarından bağımsız geçer.
- Açıkça devam isteyen satırlar bir cümle grubunda birleştirilir; her grup OpenAI Responses API ile diğer gruplardan bağımsız düzeltilir.
- Yerel LanguageTool düzeltilmiş cümleyi kontrol eder ve uygun kural düzeltmelerini uygular.
- Kod, OCR metni ile kabul edilen son cümle arasında token diff yaparak `insert`, `replace` ve `delete` işlemlerini çıkarır.
- Harf ve rakamı karıştıran şüpheli OCR okumaları otomatik kelime silme işleminden çıkarılır.
- `1 year old` ve `2+ years old` yaş uyumu ayrıca deterministik olarak kontrol edilir.
- Tarayıcıdaki Canvas API, hatalı kısmın üzerine mavi çizgi ve doğrusunu el yazısı stiliyle ekler.
- API anahtarı hiçbir zaman tarayıcıya gönderilmez.

Model koordinatları görselden tahmin ettiği için özellikle eğik, bulanık veya gölgeli fotoğraflarda yerleşim birkaç piksel sapabilir. En iyi sonuç için fotoğrafı kâğıda dik açıyla ve iyi ışıkta çekin.

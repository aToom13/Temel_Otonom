# Test Kapsamı ve Yönergeleri

## Test Dizini Yapısı
- `unit/` : Birim testler (her modül/algoritma için izole testler)
- `integration/` : Modüller arası entegrasyon ve sistem testleri

## Test Çalıştırma

```bash
pytest --cov=core --cov=modules --cov-report=term-missing
```

## Hedefler
- Tüm ana algoritmalar, modeller ve kontrolcüler için birim testler
- Web arayüzü API ve WebSocket endpoint'leri için entegrasyon testleri
- Kod kapsamı raporu (coverage)
- CI/CD entegrasyonu için örnek GitHub Actions workflow

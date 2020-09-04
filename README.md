Mọi người clone về và xóa folder imgtxtenh đi

Sau đó clone từ https://github.com/mauvilsa/imgtxtenh về

Chạy các lệnh:
1. `cd imgtxtenh`
2. `cmake -DCMAKE_BUILD_TYPE=Release`
3. `make`
4. `cd ..`
5. `sudo docker-compose up -d`

API sẽ được tạo ra ở `localhost:5000/api`

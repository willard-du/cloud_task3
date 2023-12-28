from flask import Flask, request, jsonify, render_template
import redis
import json
from azure.cosmos import CosmosClient, exceptions
import math
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import time

# password is the "Primary" copied in "Access keys"
redis_passwd = "RtiqhE1ACUSfNVMx9apVio6kBhDOLl1SvAzCaKYBwBY="
# "Host name" in properties
redis_host = "redisddw.redis.cache.windows.net"
# SSL Port
redis_port = 6380

cache = redis.StrictRedis(
    host=redis_host, port=redis_port,
    db=0, password=redis_passwd,
    ssl=True,
)

if cache.ping():
    print("pong")


app = Flask(__name__)

# Azure Cosmos DB 配置
cosmos_endpoint = "https://tutorial-uta-cse6332.documents.azure.com:443/"
cosmos_key = "fSDt8pk5P1EH0NlvfiolgZF332ILOkKhMdLY6iMS2yjVqdpWx4XtnVgBoJBCBaHA8PIHnAbFY4N9ACDbMdwaEw=="
cosmos_client = CosmosClient(cosmos_endpoint, cosmos_key)
database_name = "tutorial"
database = cosmos_client.get_database_client(database_name)
container_name_cities = "us_cities"
container_name_reviews = "reviews"

# 连接到数据库和容器
container_cities = database.get_container_client(container_name_cities)
container_reviews = database.get_container_client(container_name_reviews)


def purge_cache():
    for key in cache.keys():
        cache.delete(key.decode())


@app.route('/purge_cache', methods=['GET'])
def handle_purge_cache():
    purge_cache()
    return jsonify({"message": "Cache purged successfully"})


# 计算欧拉距离
def calculate_eular_distance(lat1, lng1, lat2, lng2):
    lat1 = float(lat1)
    lng1 = float(lng1)
    lat2 = float(lat2)
    lng2 = float(lng2)
    return math.sqrt((lat1 - lat2) ** 2 + (lng1 - lng2) ** 2)

# 计算欧氏距离
def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2) ** 2))

# 第一个功能的路由处理函数
@app.route('/stat/closest_cities', methods=['GET'])
def closest_cities():
    city_name = request.args.get('city').strip().lower()  # 将输入转换为小写并去除空格
    state_name = request.args.get('state').strip().lower()  # 同上
    page_size = int(request.args.get('page_size', 50))
    page = int(request.args.get('page', 1))
    cache_key = f"closest_cities:{city_name}:{state_name}:{page_size}:{page}"

    start_time = time.time()
    # 尝试从 Redis 获取缓存的结果
    cached_result = cache.get(cache_key)
    if cached_result:
        elapsed_time = int((time.time() - start_time) * 1000)
        return jsonify({"result": json.loads(cached_result), "from_cache": True, "compute_time": elapsed_time})

    # 从数据库获取指定城市的坐标
    query = "SELECT c.lat, c.lng FROM c WHERE LOWER(c.city) = @city AND LOWER(c.state) = @state"
    parameters = [{"name": "@city", "value": city_name}, {"name": "@state", "value": state_name}]
    city_data = list(container_cities.query_items(query=query, parameters=parameters, enable_cross_partition_query=True))

    if not city_data:
        return jsonify({"error": "City not found"}), 404

    city_lat, city_lng = city_data[0]['lat'], city_data[0]['lng']

    # 获取所有城市并计算距离
    cities = list(container_cities.read_all_items())
    cities_with_distance = []
    for city in cities:
        city_name_lower = city['city'].lower()
        state_name_lower = city['state'].lower()
        if city_name_lower != city_name or state_name_lower != state_name:  # 排除查询的城市
            city['distance'] = calculate_eular_distance(city_lat, city_lng, city['lat'], city['lng'])
            cities_with_distance.append(city)

    # 根据距离排序并实现分页
    cities_with_distance.sort(key=lambda x: x['distance'])
    start = (page - 1) * page_size
    end = start + page_size
    paginated_cities = cities_with_distance[start:end]

    # 将结果缓存到 Redis
    cache.set(cache_key, json.dumps(paginated_cities), ex=60*60)

    elapsed_time = int((time.time() - start_time) * 1000)

    return jsonify({"result": paginated_cities, "from_cache": False, "compute_time": elapsed_time})






# 计算欧氏距离
def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2) ** 2))

# 第二个功能的路由处理函数
@app.route('/stat/knn_reviews', methods=['GET'])
def knn_reviews():
    start_time = time.time()
    try:
        classes = int(request.args.get('classes', 6))
        k = int(request.args.get('k', 3))
        words = int(request.args.get('words', 100))
        cache_key = f"knn_reviews:{classes}:{k}:{words}"

        # 尝试从 Redis 获取缓存的结果
        cached_result = cache.get(cache_key)
        if cached_result:
            return jsonify({"result": json.loads(cached_result), "from_cache": True, "time_ms": int((time.time() - start_time) * 1000)})

        # 从 Cosmos DB 读取城市坐标数据
        city_coordinates = {}
        query = "SELECT c.city, c.lat, c.lng FROM c"
        cities = container_cities.query_items(query=query, enable_cross_partition_query=True)
        for city in cities:
            city_coordinates[city['city']] = (float(city['lat']), float(city['lng']))

        # 从 Cosmos DB 读取评论数据
        reviews_data = []
        query = "SELECT c.score, c.city, c.review FROM c"
        reviews = container_reviews.query_items(query=query, enable_cross_partition_query=True)
        for review in reviews:
            reviews_data.append(review)

        # 使用 TF-IDF 对评论文本进行向量化
        vectorizer = TfidfVectorizer(stop_words='english')
        X = vectorizer.fit_transform([review['review'] for review in reviews_data])

        # 使用 KMeans 算法进行聚类
        kmeans = KMeans(n_clusters=classes)
        kmeans.fit(X)

        # 将评论分配到聚类中
        clusters = [[] for _ in range(classes)]
        for idx, label in enumerate(kmeans.labels_):
            clusters[label].append(reviews_data[idx])

        cluster_results = []
        for cluster_id, cluster_reviews in enumerate(clusters):
            if not cluster_reviews:
                continue  # 如果某个聚类中没有评论，则跳过

            # 提取该聚类中所有评论的城市
            cities_in_cluster = [review['city'] for review in cluster_reviews if review['city'] in city_coordinates]
            center_city = cities_in_cluster[0] if cities_in_cluster else None

            # 如果中心城市存在，则从城市列表中移除
            if center_city and center_city in cities_in_cluster:
                cities_in_cluster.remove(center_city)

            # 确保列表中最多只有 k 个城市
            cities_in_cluster = cities_in_cluster[:k]

            # 计算最流行的词汇
            cluster_texts = [review['review'].lower() for review in cluster_reviews]
            vectorizer_cluster = TfidfVectorizer(stop_words='english')
            X_cluster = vectorizer_cluster.fit_transform(cluster_texts)
            word_freq = np.sum(X_cluster, axis=0)
            words_freq_array = np.array(word_freq).flatten()
            popular_words = [word for word, freq in
                             sorted(zip(vectorizer_cluster.get_feature_names_out(), words_freq_array), key=lambda x: x[1],
                                    reverse=True)[:words]]

            # 计算加权平均分
            weighted_scores = np.average([int(review['score']) for review in cluster_reviews],
                                         weights=[1 for _ in cluster_reviews])

            cluster_results.append({
                "center_city": center_city,
                "cities_in_cluster": cities_in_cluster,
                "popular_words": popular_words,
                "weighted_average_score": weighted_scores
            })

        # 将结果缓存到 Redis
        cache.set(cache_key, json.dumps(cluster_results), ex=60*60)  # 缓存一个小时

        # 计算响应时间
        elapsed_time = int((time.time() - start_time) * 1000)

        return jsonify({
            "result": cluster_results,
            "from_cache": False,
            "time_ms": elapsed_time
        })

    except Exception as e:
        return jsonify({"error": str(e)})


# 计算城市的平均评论评分
@app.route('/stat/city_reviews', methods=['GET'])
def city_reviews():
    city_name = request.args.get('city').strip().lower()
    state_name = request.args.get('state').strip().lower()
    page_size = int(request.args.get('page_size', 10))
    page = int(request.args.get('page', 1))
    cache_key = f"city_reviews:{city_name}:{state_name}:{page_size}:{page}"

    start_time = time.time()
    cached_result = cache.get(cache_key)
    if cached_result:
        elapsed_time = int((time.time() - start_time) * 1000)
        return jsonify({"result": json.loads(cached_result), "from_cache": True, "compute_time": elapsed_time})

    # 从数据库获取指定城市的坐标
    query = "SELECT c.lat, c.lng FROM c WHERE LOWER(c.city) = @city AND LOWER(c.state) = @state"
    parameters = [{"name": "@city", "value": city_name}, {"name": "@state", "value": state_name}]
    city_data = list(container_cities.query_items(query=query, parameters=parameters, enable_cross_partition_query=True))

    if not city_data:
        return jsonify({"error": "City not found"}), 404

    city_lat, city_lng = city_data[0]['lat'], city_data[0]['lng']

    # 从数据库获取所有评论并计算每个城市的平均评分
    all_reviews = list(container_reviews.read_all_items())
    city_scores = {}
    for review in all_reviews:
        city_lower = review['city'].lower()
        if city_lower not in city_scores:
            city_scores[city_lower] = {'total_score': 0, 'count': 0}

        # 确保评分是整数类型
        try:
            score = int(review['score'])
        except ValueError:
            # 如果评分不能转换为整数，则跳过这条评论
            continue

        city_scores[city_lower]['total_score'] += score
        city_scores[city_lower]['count'] += 1

    for key, value in city_scores.items():
        value['average_score'] = value['total_score'] / value['count']

    # 获取所有城市的位置信息并计算距离
    all_cities = list(container_cities.read_all_items())
    cities_with_scores = []
    for city in all_cities:
        city_lower = city['city'].lower()
        if city_lower in city_scores:
            distance = calculate_eular_distance(city_lat, city_lng, city['lat'], city['lng'])
            cities_with_scores.append({
                'city': city['city'],
                'state': city['state'],
                'average_score': city_scores[city_lower]['average_score'],
                'distance': distance
            })

    cities_with_scores.sort(key=lambda x: x['distance'])

    # 分页
    start = (page - 1) * page_size
    end = start + page_size
    paginated_cities = cities_with_scores[start:end]

    cache.set(cache_key, json.dumps(paginated_cities), ex=60*60*24*5)
    elapsed_time = int((time.time() - start_time) * 1000)

    return jsonify({"result": paginated_cities, "from_cache": False, "compute_time": elapsed_time})


@app.route('/stat/clustering_pie_chart', methods=['GET'])
def clustering_pie_chart():
    try:
        classes = int(request.args.get('classes', 6))
        words = int(request.args.get('words', 100))

        # 构建缓存键
        cache_key = f'clustering_pie_chart:{classes}:{words}'

        # 尝试从 Redis 获取缓存的结果
        cached_result = cache.get(cache_key)
        if cached_result:
            return jsonify(json.loads(cached_result))

        # 从 Cosmos DB 读取评论数据
        reviews_data = []
        query = "SELECT c.review FROM c"
        reviews = container_reviews.query_items(query=query, enable_cross_partition_query=True)
        for review in reviews:
            reviews_data.append(review['review'])

        # 使用 TF-IDF 对评论文本进行向量化
        vectorizer = TfidfVectorizer(max_features=words, stop_words='english')
        X = vectorizer.fit_transform(reviews_data)

        # 使用 KMeans 算法进行聚类
        kmeans = KMeans(n_clusters=classes)
        kmeans.fit(X)

        # 计算每个聚类的样本数量
        class_counts = np.bincount(kmeans.labels_, minlength=classes)

        result = {
            "class_counts": class_counts.tolist(),
            "classes": classes
        }

        # 缓存结果到 Redis
        cache.set(cache_key, json.dumps(result), ex=3600*24*5)  # 缓存5天

        return jsonify({
            "class_counts": class_counts.tolist(),
            "classes": classes
        })
    except Exception as e:
        return jsonify({"error": str(e)})


@app.route('/q10')
def query_cities():
    return render_template('index3.html')


@app.route('/q11')
def review_cities():
    return render_template('index4.html')


@app.route('/q12')
def index():
    return render_template('index.html')
# Dunwei Du 76004 & Qingwei Zeng 76028


if __name__ == '__main__':
    app.run(debug=True)


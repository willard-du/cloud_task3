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
    city_name = request.args.get('city')
    page_size = int(request.args.get('page_size', 50))
    page = int(request.args.get('page', 0))
    cache_key = f"closest_cities:{city_name}:{page_size}:{page}"

    start_time = time.time()
    # 尝试从 Redis 获取缓存的结果
    cached_result = cache.get(cache_key)
    if cached_result:
        elapsed_time = int((time.time() - start_time) * 1000)
        return jsonify({"result": json.loads(cached_result), "from_cache": True, "compute_time": elapsed_time})

    # 从数据库获取指定城市的坐标
    query = "SELECT c.lat, c.lng FROM c WHERE c.city = @city"
    parameters = [{"name": "@city", "value": city_name}]
    city_data = list(container_cities.query_items(query=query, parameters=parameters, enable_cross_partition_query=True))

    if not city_data:
        return jsonify({"error": "City not found"}), 404

    city_lat, city_lng = city_data[0]['lat'], city_data[0]['lng']

    # 获取所有城市并计算欧拉距离
    cities = list(container_cities.read_all_items())
    for city in cities:
        city['distance'] = calculate_eular_distance(city_lat, city_lng, city['lat'], city['lng'])

    # 根据距离排序并实现分页
    cities.sort(key=lambda x: x['distance'])
    start = page * page_size
    end = start + page_size
    paginated_cities = cities[start:end]

    # 将结果缓存到 Redis
    cache.set(cache_key, json.dumps(paginated_cities), ex=60*60)  # 缓存一个小时

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

@app.route('/')
def index():
    return render_template('index2.html')
# Dunwei Du 76004 & Qingwei Zeng 76028
if __name__ == '__main__':
    app.run(debug=True)


from utils import create_spark_session
spark = create_spark_session('inspect')
books = spark.read.option('header', True).option('inferSchema', True).csv('outputs/tables/books_clean_csv')
print('columns=', books.columns)
if 'publication_year' in books.columns:
    pdf = books.select('publication_year').toPandas()
    vals = pdf['publication_year'].astype(str).head(50).tolist()
    print('sample publication_year values:', vals)
    bad = [v for v in pdf['publication_year'].astype(str).unique() if any(ord(ch) < 32 for ch in v)]
    print('bad count:', len(bad))
    print('bad values sample:', bad[:20])
else:
    print('no publication_year column')
spark.stop()

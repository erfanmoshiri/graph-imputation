import snowflake.connector
import credentials
from snowflake.connector.pandas_tools import write_pandas

from sqlalchemy import create_engine
# from snowflake.sqlalchemy import URL
def query_snowflake_pd(query, database="PRODDOMAINDW", role="data_dept_development"):
    domain_dw_conn = snowflake.connector.connect(user=credentials.snowflake_username,
                                                 account="re61407.ap-southeast-2",
                                                 warehouse="WAREHOUSE_XS", database=database, role=role,
                                                 authenticator="externalbrowser",
                                                 autocommit=True)

    # domain_dw_conn = snowflake.connector.connect(
    #     user=credentials.snowflake_username,
    #     account=credentials.snowflake_password,
    #     region=REGION,
    #     warehouse=WAREHOUSE,
    #     database=DATABASE,
    #     schema=SCHEMA,
    #     role=ROLE,
    #     authenticator="externalbrowser",
    #     autocommit=True)
    domain_dw_cursor = domain_dw_conn.cursor()

    domain_dw_cursor.execute(query)
    result = domain_dw_cursor.fetch_pandas_all()
    domain_dw_cursor.close()
    domain_dw_conn.close()
    return result

def write_snowflake_pd(df, name, schema, db, role="ROLE_PROPERTY_DATA_RESTRICTED"):
    domain_dw_conn = snowflake.connector.connect(user=credentials.snowflake_username, password=credentials.snowflake_password,
                                                 account="re61407.ap-southeast-2",
                                                 warehouse="WAREHOUSE_XS", database=db, schema=schema, role=role,
                                                 authenticator="externalbrowser",
                                                 autocommit=True
                                                 )
    # df.to_sql(name,domain_dw_conn,schema, index=index, if_exists=if_exists, method=pd_writer)
    df.columns = [c.upper() for c in df.columns]
    success, num_chunks, num_rows, output = write_pandas(
        conn=domain_dw_conn,
        df=df,
        table_name=name,
        database=db,
        schema=schema
    )
    domain_dw_conn.close()

def execute_snowflake(query):
    domain_dw_conn = snowflake.connector.connect(user=credentials.snowflake_username,
                                                 password=credentials.snowflake_password,
                                                 account="re61407.ap-southeast-2",
                                                 warehouse="WAREHOUSE_XS", database="PRODDOMAINDW", role="ROLE_PROPERTY_DATA_RESTRICTED",
                                                 authenticator="externalbrowser",
                                                 autocommit=True
                                                 )
    domain_dw_cursor = domain_dw_conn.cursor()
    domain_dw_cursor.execute(query)
    domain_dw_cursor.close()
    domain_dw_conn.close()

def get_snowflake_engine(db, schema, role="ROLE_PROPERTY_DATA_RESTRICTED"):
    account_identifier = "re61407.ap-southeast-2"
    user = credentials.snowflake_username
    password = credentials.snowflake_password
    conn_string = f"snowflake://{user}:{password}@{account_identifier}/{db}/{schema}?warehouse=WAREHOUSE_XS&role={role}"
    engine = create_engine(conn_string)
    return engine

def create_table_from_df(df, name, schema, db, role="ROLE_PROPERTY_DATA_RESTRICTED", if_exists='replace'):
    engine = get_snowflake_engine(db, schema, role)
    # connection = engine.raw_connection()
    with engine.connect() as conn:
        df.iloc[0:10].to_sql(name, conn, index=False, if_exists=if_exists, method='multi')
    execute_snowflake(f'''truncate table {db}.{schema}.{name}''')
CREATE TABLE retail_sales (
    date DATE NOT NULL,
    store_id VARCHAR(10) NOT NULL,
    sku_id VARCHAR(20) NOT NULL,
    category VARCHAR(50) NOT NULL,
    region VARCHAR(30) NOT NULL,
    price DECIMAL(10,2) NOT NULL,
    promotion INTEGER NOT NULL,
    holiday INTEGER NOT NULL,
    units_sold INTEGER NOT NULL,
    lead_time_days INTEGER NOT NULL
);

CREATE INDEX idx_retail_sales_store_sku_date
ON retail_sales(store_id, sku_id, date);

-- 1) Daily demand by store and SKU
SELECT date, store_id, sku_id, SUM(units_sold) AS daily_units
FROM retail_sales
GROUP BY date, store_id, sku_id
ORDER BY date, store_id, sku_id;

-- 2) Promotion lift by SKU
SELECT
    sku_id,
    AVG(CASE WHEN promotion = 1 THEN units_sold END) AS promo_avg_units,
    AVG(CASE WHEN promotion = 0 THEN units_sold END) AS regular_avg_units,
    AVG(CASE WHEN promotion = 1 THEN units_sold END)
      - AVG(CASE WHEN promotion = 0 THEN units_sold END) AS absolute_lift
FROM retail_sales
GROUP BY sku_id
ORDER BY absolute_lift DESC;

-- 3) 7-day rolling demand using a window function
SELECT
    date,
    store_id,
    sku_id,
    units_sold,
    AVG(units_sold) OVER (
        PARTITION BY store_id, sku_id
        ORDER BY date
        ROWS BETWEEN 6 PRECEDING AND CURRENT ROW
    ) AS rolling_7d_units
FROM retail_sales;

-- 4) High-volume SKUs by region
WITH sku_region AS (
    SELECT region, sku_id, SUM(units_sold) AS total_units
    FROM retail_sales
    GROUP BY region, sku_id
), ranked AS (
    SELECT *, ROW_NUMBER() OVER (PARTITION BY region ORDER BY total_units DESC) AS rn
    FROM sku_region
)
SELECT region, sku_id, total_units
FROM ranked
WHERE rn <= 5
ORDER BY region, rn;

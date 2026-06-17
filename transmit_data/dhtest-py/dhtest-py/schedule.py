from datetime import datetime, timedelta

def get_previous_minute_data(client):
    end_time = datetime.utcnow()
    start_time = end_time - timedelta(seconds=10)

    start_time_str = start_time.isoformat() + "Z"
    end_time_str = end_time.isoformat() + "Z"

    print(f"Querying data from InfluxDB between {start_time_str} and {end_time_str}")

    try:
        query = f"""
        SELECT 
            MIN("value") AS min_value, 
            MAX("value") AS max_value, 
            MEAN("value") AS avg_value 
        FROM "wave_data" 
        WHERE time >= '{start_time_str}' AND time < '{end_time_str}'
        GROUP BY "channel"
        """
        result = client.query(query)

        query2 = f"""
        SELECT * FROM "wave_data"
        ORDER BY time DESC
        LIMIT 1
        """
        result2 = client.query(query2)

        # 返回的结果是一个字典，每个 (measurement, tag set) 对应一个 series
        grouped_results = {}
        print("result",result)
        print("result2:",result2)

        for (measurement, tags), points in result.items():
            channel = tags.get("channel")
            for point in points:
                if(point.get("min_value") is not None or point.get("max_value") is not None or point.get("avg_value") is not None):
                    grouped_results[channel] = {
                        "time": point.get("time"),
                    }
                if point.get("min_value") is not None:  # 如果这一组有数据
                    grouped_results[channel]['2'] = point.get('min_value')
                if point.get("max_value") is not None:
                    grouped_results[channel]['1'] = point.get('max_value')
                if point.get("avg_value") is not None:
                    grouped_results[channel]['3'] = point.get('avg_value')

        print("Grouped results:", grouped_results)
        return grouped_results

    except Exception as e:
        print(f"Failed to query InfluxDB: {e}")
        return {}

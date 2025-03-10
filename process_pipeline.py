def process_dataset(dataset, processors, batch_size=5000, load_cache=True):    
    # ✅ 直接用 `map()` 处理整个 `dataset`
    for processor in processors:
        print(f"processor: {processor['processor_function'].__name__}")
        dataset = dataset.map(
            lambda batch: processor['processor_function'](batch),
            batched=True,
            batch_size=batch_size,
            num_proc= processor['num_proc'],
            load_from_cache_file=load_cache
        )        
    return dataset


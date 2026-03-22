package com.swimming.services;

import com.swimming.models.StrokeMetric;
import com.swimming.repositories.StrokeMetricRepository;
import org.springframework.stereotype.Service;

import java.util.List;

@Service
public class StrokeMetricService {

    private final StrokeMetricRepository repository;

    public StrokeMetricService(StrokeMetricRepository repository) {
        this.repository = repository;
    }

    public List<StrokeMetric> getAll() {
        return repository.findAll();
    }

    public StrokeMetric save(StrokeMetric metric) {
        return repository.save(metric);
    }
}
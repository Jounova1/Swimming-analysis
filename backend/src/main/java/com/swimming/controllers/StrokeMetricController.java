package com.swimming.controllers;

import java.util.List;

import com.swimming.models.StrokeMetric;
import com.swimming.services.StrokeMetricService;

import org.springframework.web.bind.annotation.*;

@RestController
@RequestMapping("/metrics")
public class StrokeMetricController 
{

    private final StrokeMetricService service;

    public StrokeMetricController(StrokeMetricService service)
 {
        this.service = service;
    }

  
    @GetMapping
    public List<StrokeMetric> getAllMetrics() 
    {
        return service.getAll();
    }

   
    @PostMapping
    public StrokeMetric save(@RequestBody StrokeMetric metric) 
    {
        return service.save(metric);
    }
}
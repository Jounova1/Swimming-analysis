package com.swimming.controllers;

import com.swimming.models.StrokeMetric;
import com.swimming.services.AnalysisService;
import org.springframework.web.bind.annotation.*;

@RestController
@RequestMapping("/analysis")
public class AnalysisController {

    private final AnalysisService service;

    public AnalysisController(AnalysisService service) {
        this.service = service;
    }

   
    @PostMapping("/{sessionId}")
    public StrokeMetric analyze(@PathVariable Long sessionId,
                                @RequestBody AnalysisRequest request) {

        return service.saveAnalysis(
                sessionId,
                request.getStrokeRate(),
                request.getVelocity()
        );
    }
}
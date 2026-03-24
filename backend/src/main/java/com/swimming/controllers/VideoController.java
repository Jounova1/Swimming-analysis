package com.swimming.controllers;

import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

import com.swimming.models.Video;
import com.swimming.services.VideoService;

@RestController
@RequestMapping("/video")

public class VideoController {
    
    private VideoService service;

    public VideoController(VideoService service)
    {
        this.service=service;
    }

    @PostMapping("/{Id}")
    public Video Upload(@PathVariable Long Id,@RequestBody Video video)
    {
        return service.saveVideo(video);
    }
}

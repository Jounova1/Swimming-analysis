package com.swimming.services;

<<<<<<< HEAD
public class VideoService {

=======
import java.util.List;

import org.springframework.stereotype.Service;

import com.swimming.models.Video;
import com.swimming.repositories.VideoRepository;

@Service
public class VideoService {

    private final VideoRepository videoRepository;

    public VideoService(VideoRepository videoRepository) {
        this.videoRepository = videoRepository;
    }

    public List<Video> getAllVideos() {
        return videoRepository.findAll();
    }

    public Video getVideoById(Long id) {
        return videoRepository.findById(id).orElseThrow(() -> new RuntimeException("Video not found"));
    }

    public Video saveVideo(Video video) {
        return videoRepository.save(video);
    }

    public void deleteVideo(Long id) {
        videoRepository.deleteById(id);
    }
>>>>>>> 746d4a46a164173f06d1e85d80b3f9565be05583
}

package com.swimming.services;

import java.util.List;

import org.springframework.stereotype.Service;

import com.swimming.models.Swimmer;
import com.swimming.repositories.SwimmerRepository;

@Service
public class SwimmerService {

    private final SwimmerRepository swimmerRepository;

    public SwimmerService(SwimmerRepository swimmerRepository) {
        this.swimmerRepository = swimmerRepository;
    }

    public List<Swimmer> getAllSwimmers() {
        return swimmerRepository.findAll();
    }

    public Swimmer getSwimmerById(Long id) {
        return swimmerRepository.findById(id).orElse(null);
    }

    public Swimmer createSwimmer(Swimmer swimmer) {
        return swimmerRepository.save(swimmer);
    }

    public Swimmer updateSwimmer(Long id, Swimmer swimmerDetails) {
        if (swimmerRepository.existsById(id)) {
            return swimmerRepository.save(swimmerDetails);
        }
        return null;
    }

    public void deleteSwimmer(Long id) {
        swimmerRepository.deleteById(id);
    }

    public void saveSwimmer(Swimmer swimmer) {
        swimmerRepository.save(swimmer);
    }
}
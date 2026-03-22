package com.swimming.services;

import java.util.List;

import org.springframework.stereotype.Service;

import com.swimming.models.Session;
import com.swimming.models.Swimmer;
import com.swimming.repositories.SessionRepository;
import com.swimming.repositories.SwimmerRepository;

@Service
public class SessionService {
    private final SessionRepository sessionRepository;
    private final SwimmerRepository swimmerRepository;

    public SessionService(SessionRepository sessionRepository, SwimmerRepository swimmerRepository) {
        this.sessionRepository = sessionRepository;
        this.swimmerRepository = swimmerRepository;
    }

    public List<Session> getAllSessions() {
        return sessionRepository.findAll();
    }

    public Session getSessionById(Long id) {
        return sessionRepository
                .findById(id)
                .orElseThrow(() -> new RuntimeException("Session not found"));
    }

    public Session createSession(Long swimmerId, Session session) {
        Swimmer swimmer = swimmerRepository.findById(swimmerId)
                .orElseThrow(() -> new RuntimeException("Swimmer not found"));

        // If Session has a swimmer field, you may want session.setSwimmer(swimmer);
        return sessionRepository.save(session);
    }

    public Session createSession(Session session) {
        return sessionRepository.save(session);
    }

    public Session updateSession(Long id, Session sessionDetails) {
        if (!sessionRepository.existsById(id)) {
            throw new RuntimeException("Session not found");
        }
        sessionDetails.setId(id);
        return sessionRepository.save(sessionDetails);
    }

    public void deleteSession(Long id) {
        sessionRepository.deleteById(id);
    }


package com.swimming.services;

import java.util.List;

import com.swimming.models.Session;
import com.swimming.repositories.SessionRepository;

public class SessionService {
    private SessionRepository sessionRepository;

    public SessionService(SessionRepository sessionRepository) {
        this.sessionRepository = sessionRepository;
    }

    public List<Session> getAllSessions() {
        return sessionRepository.findAll();
    }

    public Session getSessionById(Long id) {
        return sessionRepository.findById(id).orElse(null);
    }

    public Session createSession(Session session) {
        return sessionRepository.save(session);
    }

    public Session updateSession(Long id, Session sessionDetails) {
        Session session = sessionRepository.findById(id).orElse(null);
        if (session != null) {
            return sessionRepository.save(session);
        }
        return null;
    }

    public void deleteSession(Long id) {
        sessionRepository.deleteById(id);
    }
}
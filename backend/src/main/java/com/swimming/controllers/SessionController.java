package com.swimming.controllers;

import java.util.List;

import org.springframework.web.bind.annotation.DeleteMapping;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.PostMapping;

import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

import com.swimming.models.Session;
import com.swimming.services.SessionService;

@RestController
@RequestMapping("/session")

public class SessionController {
    private SessionService service;

    public SessionController(SessionService service)
    {
        this.service=service;
    }

    @GetMapping
    public List<Session> FindAllSessions()
    {
        return service.getAllSessions();
    }

    @DeleteMapping("/{Id}")
    public void DeleteSessions (@PathVariable Long Id)
    {
        service.deleteSession(Id);
    }
    @PostMapping("/{Id}")
    public Session AddSession (@PathVariable Long Id,@RequestBody Session session)
    {
        return service.createSession(Id, session);
    }

}

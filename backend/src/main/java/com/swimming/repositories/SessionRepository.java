package com.swimming.repositories;

import org.springframework.data.jpa.repository.JpaRepository;

import com.swimming.models.Session;

public interface SessionRepository extends JpaRepository <Session,Long>
{
    
}

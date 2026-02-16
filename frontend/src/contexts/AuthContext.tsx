'use client';

import React, { createContext, useContext, useState, useEffect, useCallback, useMemo, ReactNode } from 'react';
import { User } from 'firebase/auth';
import {
  signUp as firebaseSignUp,
  signIn as firebaseSignIn,
  signOut as firebaseSignOut,
  onAuthChange,
  getUserProfile,
  UserProfile,
} from '@/services/firebaseService';

interface AuthContextType {
  user: User | null;
  profile: UserProfile | null;
  loading: boolean;
  signUp: (email: string, password: string, displayName: string) => Promise<void>;
  signIn: (email: string, password: string) => Promise<void>;
  signOut: () => Promise<void>;
}

const AuthContext = createContext<AuthContextType | undefined>(undefined);

export function AuthProvider({ children }: { children: ReactNode }) {
  const [user, setUser] = useState<User | null>(null);
  const [profile, setProfile] = useState<UserProfile | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const unsubscribe = onAuthChange(async (firebaseUser) => {
      setUser(firebaseUser);

      if (firebaseUser) {
        try {
          const userProfile = await getUserProfile(firebaseUser.uid);
          setProfile(userProfile);
        } catch (error) {
          console.error('[AuthContext] Failed to load user profile:', error);
          setProfile(null);
        }
      } else {
        setProfile(null);
      }

      setLoading(false);
    });

    return unsubscribe;
  }, []);

  const signUp = useCallback(async (email: string, password: string, displayName: string) => {
    await firebaseSignUp(email, password, displayName);
  }, []);

  const signIn = useCallback(async (email: string, password: string) => {
    await firebaseSignIn(email, password);
  }, []);

  const signOut = useCallback(async () => {
    await firebaseSignOut();
  }, []);

  const value = useMemo(() => ({
    user,
    profile,
    loading,
    signUp,
    signIn,
    signOut,
  }), [user, profile, loading, signUp, signIn, signOut]);

  return (
    <AuthContext.Provider value={value}>
      {children}
    </AuthContext.Provider>
  );
}

export function useAuth() {
  const context = useContext(AuthContext);
  if (context === undefined) {
    throw new Error('useAuth must be used within an AuthProvider');
  }
  return context;
}

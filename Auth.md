# Authentication Guide for CinemaAI

This document outlines the authentication flow and implementation details for the CinemaAI application.

## Backend Changes

The backend has been migrated from SQLite to Supabase with the following key changes:

1. **New Dependencies**:
   - `supabase` - For Supabase client
   - `python-jose` - For JWT handling
   - `passlib[bcrypt]` - For password hashing
   - `python-multipart` - For form handling

2. **New Files**:
   - `supabase_client.py` - Handles Supabase client initialization
   - `auth.py` - Contains authentication logic and middleware

3. **Updated Files**:
   - `main.py` - Added authentication endpoints
   - `config.py` - Added Supabase and JWT configurations
   - `recommender.py` - Updated to use Supabase instead of SQLite

## Authentication Flow

### 1. Sign Up
- **Endpoint**: `POST /auth/signup`
- **Request Body**:
  ```json
  {
    "email": "user@example.com",
    "password": "your-secure-password",
    "full_name": "John Doe"
  }
  ```
- **Response**:
  ```json
  {
    "access_token": "jwt_token_here",
    "token_type": "bearer",
    "user": {
      "id": "user_uuid",
      "email": "user@example.com",
      "user_metadata": {
        "full_name": "John Doe"
      }
    }
  }
  ```

### 2. Login
- **Endpoint**: `POST /auth/login`
- **Request Body**:
  ```json
  {
    "email": "user@example.com",
    "password": "securepassword123"
  }
  ```
- **Response**: Same as signup response

### 3. Get Current User
- **Endpoint**: `GET /auth/me`
- **Headers**:
  ```
  Authorization: Bearer your_jwt_token_here
  ```
- **Response**: User object

## Frontend Implementation (Next.js)

### 1. Install Dependencies
```bash
npm install @supabase/supabase-js @supabase/auth-helpers-nextjs
```

### 2. Create Supabase Client
Create `lib/supabaseClient.js`:
```javascript
import { createClient } from '@supabase/supabase-js'

const supabaseUrl = process.env.NEXT_PUBLIC_SUPABASE_URL
const supabaseAnonKey = process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY

export const supabase = createClient(supabaseUrl, supabaseAnonKey)
```

### 3. Create Auth Context
Create `context/AuthContext.js`:
```jsx
import { createContext, useContext, useEffect, useState } from 'react'
import { supabase } from '@/lib/supabaseClient'

const AuthContext = createContext({})

export const useAuth = () => useContext(AuthContext)

export const AuthProvider = ({ children }) => {
  const [user, setUser] = useState(null)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    // Check active sessions and set the user
    const session = supabase.auth.session()
    setUser(session?.user ?? null)
    
    // Listen for changes in auth state
    const { data: listener } = supabase.auth.onAuthStateChange(
      async (event, session) => {
        setUser(session?.user ?? null)
        setLoading(false)
      }
    )

    return () => {
      listener?.unsubscribe()
    }
  }, [])

  // Sign up with email and password
  const signUp = async (email, password, fullName) => {
    const { user, error } = await supabase.auth.signUp(
      {
        email,
        password,
      },
      {
        data: {
          full_name: fullName,
        },
      }
    )
    return { user, error }
  }

  // Sign in with email and password
  const signIn = async (email, password) => {
    const { user, error } = await supabase.auth.signInWithPassword({
      email,
      password,
    })
    return { user, error }
  }

  // Sign out
  const signOut = async () => {
    await supabase.auth.signOut()
  }

  const value = {
    signUp,
    signIn,
    signOut,
    user,
    loading,
  }

  return (
    <AuthContext.Provider value={value}>
      {!loading && children}
    </AuthContext.Provider>
  )
}
```

### 4. Create Auth Components

#### Sign Up Component (`components/auth/SignUp.js`)
```jsx
import { useState } from 'react'
import { useRouter } from 'next/router'
import { useAuth } from '@/context/AuthContext'

export default function SignUp() {
  const [email, setEmail] = useState('')
  const [password, setPassword] = useState('')
  const [fullName, setFullName] = useState('')
  const [error, setError] = useState('')
  const [loading, setLoading] = useState(false)
  const { signUp } = useAuth()
  const router = useRouter()

  const handleSubmit = async (e) => {
    e.preventDefault()
    setError('')
    setLoading(true)
    
    try {
      const { error } = await signUp(email, password, fullName)
      if (error) throw error
      router.push('/dashboard')
    } catch (error) {
      setError(error.message)
      setLoading(false)
    }
  }

  return (
    <div className="max-w-md mx-auto mt-10">
      <h1 className="text-2xl font-bold mb-6">Sign Up</h1>
      {error && (
        <div className="bg-red-100 border border-red-400 text-red-700 px-4 py-3 rounded mb-4">
          {error}
        </div>
      )}
      <form onSubmit={handleSubmit} className="space-y-4">
        <div>
          <label className="block text-sm font-medium mb-1">Full Name</label>
          <input
            type="text"
            value={fullName}
            onChange={(e) => setFullName(e.target.value)}
            className="w-full p-2 border rounded"
            required
          />
        </div>
        <div>
          <label className="block text-sm font-medium mb-1">Email</label>
          <input
            type="email"
            value={email}
            onChange={(e) => setEmail(e.target.value)}
            className="w-full p-2 border rounded"
            required
          />
        </div>
        <div>
          <label className="block text-sm font-medium mb-1">Password</label>
          <input
            type="password"
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            className="w-full p-2 border rounded"
            required
          />
        </div>
        <button
          type="submit"
          disabled={loading}
          className="w-full bg-blue-500 text-white p-2 rounded hover:bg-blue-600 disabled:opacity-50"
        >
          {loading ? 'Creating Account...' : 'Sign Up'}
        </button>
      </form>
      <p className="mt-4 text-sm text-gray-600">
        Already have an account?{' '}
        <a href="/login" className="text-blue-500 hover:underline">
          Log In
        </a>
      </p>
    </div>
  )
}
```

#### Login Component (`components/auth/Login.js`)
```jsx
import { useState } from 'react'
import { useRouter } from 'next/router'
import { useAuth } from '@/context/AuthContext'

export default function Login() {
  const [email, setEmail] = useState('')
  const [password, setPassword] = useState('')
  const [error, setError] = useState('')
  const [loading, setLoading] = useState(false)
  const { signIn } = useAuth()
  const router = useRouter()

  const handleSubmit = async (e) => {
    e.preventDefault()
    setError('')
    setLoading(true)
    
    try {
      const { error } = await signIn(email, password)
      if (error) throw error
      router.push('/dashboard')
    } catch (error) {
      setError('Failed to log in. Please check your credentials.')
      setLoading(false)
    }
  }

  return (
    <div className="max-w-md mx-auto mt-10">
      <h1 className="text-2xl font-bold mb-6">Log In</h1>
      {error && (
        <div className="bg-red-100 border border-red-400 text-red-700 px-4 py-3 rounded mb-4">
          {error}
        </div>
      )}
      <form onSubmit={handleSubmit} className="space-y-4">
        <div>
          <label className="block text-sm font-medium mb-1">Email</label>
          <input
            type="email"
            value={email}
            onChange={(e) => setEmail(e.target.value)}
            className="w-full p-2 border rounded"
            required
          />
        </div>
        <div>
          <label className="block text-sm font-medium mb-1">Password</label>
          <input
            type="password"
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            className="w-full p-2 border rounded"
            required
          />
        </div>
        <button
          type="submit"
          disabled={loading}
          className="w-full bg-blue-500 text-white p-2 rounded hover:bg-blue-600 disabled:opacity-50"
        >
          {loading ? 'Logging In...' : 'Log In'}
        </button>
      </form>
      <p className="mt-4 text-sm text-gray-600">
        Don't have an account?{' '}
        <a href="/signup" className="text-blue-500 hover:underline">
          Sign Up
        </a>
      </p>
    </div>
  )
}
```

## Environment Variables

Create a `.env.local` file in your Next.js project root with:

```env
NEXT_PUBLIC_SUPABASE_URL=your_supabase_url
NEXT_PUBLIC_SUPABASE_ANON_KEY=your_supabase_anon_key
```

## Protected Routes

Create a protected route component (`components/ProtectedRoute.js`):

```jsx
import { useEffect } from 'react'
import { useRouter } from 'next/router'
import { useAuth } from '@/context/AuthContext'

export default function ProtectedRoute({ children }) {
  const { user, loading } = useAuth()
  const router = useRouter()

  useEffect(() => {
    if (!loading && !user) {
      router.push('/login')
    }
  }, [user, loading, router])

  if (loading || !user) {
    return <div>Loading...</div>
  }

  return children
}
```

## Usage in Pages

### Protected Page Example (`pages/dashboard.js`)
```jsx
import { useAuth } from '@/context/AuthContext'
import ProtectedRoute from '@/components/ProtectedRoute'

function Dashboard() {
  const { user, signOut } = useAuth()

  return (
    <ProtectedRoute>
      <div className="p-6">
        <h1>Welcome, {user?.email}!</h1>
        <button 
          onClick={signOut}
          className="mt-4 bg-red-500 text-white px-4 py-2 rounded"
        >
          Sign Out
        </button>
      </div>
    </ProtectedRoute>
  )
}

export default Dashboard
```

## Error Handling

- Handle network errors
- Show loading states
- Display user-friendly error messages
- Implement password reset functionality
- Add email verification flow

## Security Considerations

- Use HTTPS in production
- Implement rate limiting
- Add CSRF protection
- Set secure and httpOnly flags for cookies
- Implement proper CORS policies

## Testing

1. Test sign up with valid/invalid credentials
2. Test login with valid/invalid credentials
3. Test protected routes
4. Test session persistence
5. Test error handling

## Troubleshooting

- **CORS Issues**: Ensure proper CORS configuration in Supabase
- **Auth State Not Updating**: Check if the AuthProvider wraps your app
- **Session Not Persisting**: Verify localStorage/sessionStorage is enabled
- **Network Errors**: Check if Supabase URL and keys are correct

## Next Steps

- Add social login (Google, GitHub, etc.)
- Implement password reset flow
- Add email verification
- Set up 2FA (Two-Factor Authentication)
- Implement role-based access control

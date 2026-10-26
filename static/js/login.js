/**
 * login.js
 * Handles Admin Authentication interactions:
 * - Password Show / Hide toggle
 * - Client-side input validation
 * - Dynamic asynchronous login request with Loading & Error states
 * - Demo credentials quick filler
 */

document.addEventListener('DOMContentLoaded', () => {
  const loginForm = document.getElementById('login-form');
  const usernameInput = document.getElementById('username');
  const passwordInput = document.getElementById('password');
  const rememberMeCheckbox = document.getElementById('remember_me');
  const btnTogglePassword = document.getElementById('btn-toggle-password');
  const eyeIcon = document.getElementById('eye-icon');
  const eyeSlashIcon = document.getElementById('eye-slash-icon');
  const btnSubmit = document.getElementById('btn-login-submit');
  const btnText = document.getElementById('btn-text');
  const loginSpinner = document.getElementById('login-spinner');
  const arrowIcon = document.getElementById('arrow-icon');
  const alertBox = document.getElementById('login-alert');
  const alertText = document.getElementById('alert-text');
  const btnFillDemo = document.getElementById('btn-fill-demo');

  // 1. Password Visibility Toggle
  if (btnTogglePassword && passwordInput) {
    btnTogglePassword.addEventListener('click', (e) => {
      e.preventDefault();
      const isPassword = passwordInput.getAttribute('type') === 'password';
      passwordInput.setAttribute('type', isPassword ? 'text' : 'password');
      
      if (isPassword) {
        eyeIcon.style.display = 'none';
        eyeSlashIcon.style.display = 'block';
        btnTogglePassword.setAttribute('aria-label', 'Hide password');
      } else {
        eyeIcon.style.display = 'block';
        eyeSlashIcon.style.display = 'none';
        btnTogglePassword.setAttribute('aria-label', 'Show password');
      }
      passwordInput.focus();
    });
  }

  // 2. Quick Demo Fill Button
  if (btnFillDemo) {
    btnFillDemo.addEventListener('click', () => {
      usernameInput.value = 'admin';
      passwordInput.value = 'admin123';
      clearAlert();
      usernameInput.classList.remove('input-error');
      passwordInput.classList.remove('input-error');
    });
  }

  // Helper: Display Alert
  function showAlert(message, type = 'danger') {
    alertText.textContent = message;
    alertBox.className = `alert-box alert-${type} show shake`;
    
    // Remove shake class after animation completes so it can re-trigger on subsequent clicks
    setTimeout(() => {
      alertBox.classList.remove('shake');
    }, 450);
  }

  // Helper: Clear Alert
  function clearAlert() {
    alertBox.className = 'alert-box';
    alertText.textContent = '';
  }

  // Clear errors when typing
  [usernameInput, passwordInput].forEach((input) => {
    input.addEventListener('input', () => {
      input.classList.remove('input-error');
      if (!alertBox.classList.contains('alert-info')) {
        clearAlert();
      }
    });
  });

  // 3. Form Submission with Asynchronous Fetch & State Handling
  loginForm.addEventListener('submit', async (e) => {
    e.preventDefault();

    const username = usernameInput.value.trim();
    const password = passwordInput.value;
    const rememberMe = rememberMeCheckbox ? rememberMeCheckbox.checked : false;

    // Validation: Empty credentials check
    if (!username && !password) {
      usernameInput.classList.add('input-error');
      passwordInput.classList.add('input-error');
      showAlert('Please enter both username/email and password.');
      usernameInput.focus();
      return;
    }
    if (!username) {
      usernameInput.classList.add('input-error');
      showAlert('Username or email is required.');
      usernameInput.focus();
      return;
    }
    if (!password) {
      passwordInput.classList.add('input-error');
      showAlert('Password is required.');
      passwordInput.focus();
      return;
    }

    // Set Loading State
    btnSubmit.disabled = true;
    btnText.textContent = 'Authenticating...';
    loginSpinner.style.display = 'inline-block';
    arrowIcon.style.display = 'none';
    clearAlert();

    try {
      const response = await fetch('/login', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Accept': 'application/json',
        },
        body: JSON.stringify({
          username: username,
          password: password,
          remember_me: rememberMe,
        }),
      });

      const result = await response.json().catch(() => null);

      if (response.ok && result && result.success) {
        // Successful Login State
        btnText.textContent = 'Access Granted!';
        btnSubmit.style.background = 'linear-gradient(135deg, #10B981 0%, #059669 100%)';
        showAlert('Authentication verified. Redirecting to Admin Dashboard...', 'success');

        setTimeout(() => {
          window.location.href = result.redirect_url || '/dashboard';
        }, 600);
      } else {
        // Failed Login State (Invalid credentials)
        const errorMsg = (result && result.message) ? result.message : 'Invalid username or password.';
        showAlert(errorMsg, 'danger');
        usernameInput.classList.add('input-error');
        passwordInput.classList.add('input-error');

        // Reset Button
        btnSubmit.disabled = false;
        btnText.textContent = 'Sign In to Dashboard';
        loginSpinner.style.display = 'none';
        arrowIcon.style.display = 'inline-block';
        passwordInput.focus();
      }
    } catch (err) {
      // Network or Fallback error: Fall back to native form submission
      console.warn('AJAX login failed, falling back to standard POST:', err);
      loginForm.submit();
    }
  });
});

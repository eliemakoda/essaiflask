<!DOCTYPE html>
<html>
<head>
    <title>Taal Lake</title>
    <link rel="stylesheet" href="https://unpkg.com/leaflet@1.7.1/dist/leaflet.css" />
<!-- Custom fonts for this template-->
<link href="static/css/all.min.css" rel="stylesheet" type="text/css">
<link
    href="https://fonts.googleapis.com/css?family=Nunito:200,200i,300,300i,400,400i,600,600i,700,700i,800,800i,900,900i"
    rel="stylesheet">

<!-- Custom styles for this template-->
<link href="static/css/sb-admin-2.min.css" rel="stylesheet">
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap@4.3.1/dist/css/bootstrap.min.css" integrity="sha384-ggOyR0iXCbMQv3Xipma34MD+dH/1fQ784/j6cY/iJTQUOhcWr7x9JvoRxT2MZw1T" crossorigin="anonymous">
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.7.1/dist/leaflet.css" />

<link rel="stylesheet" href="https://cdn.datatables.net/1.13.3/css/jquery.dataTables.min.css">
</head>
<body>
    <div class="container">
        <!-- Outer Row -->
        <div class="row justify-content-center">
            <div class="col-xl-6 col-lg-6 col-md-6">
                <div class="card o-hidden border-0 shadow-lg my-5">
                    <div class="card-body p-0">
                        <!-- Nested Row within Card Body -->
                        <div class="row">
                            <div class="col-lg-12">
                                <div class="p-5">
                                    <div style="text-align: center;">
                                        <img src="../admin/img/logo1.png" alt="LLDA Logo" style="width:50%;">
                                    </div>
                                    <div class="text-center">
                                        <h1 class="h7 text-gray-900 mb-4" style="font-size: 40px; font-family: fantasy; padding-top: 20px;">LAGUNA LAKE DEVELOPMENT AUTHORITY</h1>
                                    </div>
                                    <hr>
                                    <div class="text-center">
                                        <h1 class="h4 text-gray-900 mb-4">ADMINISTRATOR</h1>
                                    </div>
                                    <form class="user" action="code.php" method="POST">
                                        
                                        <div class="form-group">
                                            <input type="username" name="username" class="form-control form-control-user" placeholder="Enter Username...">
                                        </div>
                                        <div class="form-group">
                                            <input type="password" name="password" class="form-control form-control-user" placeholder="Enter Password">
                                        </div>
                                        <button type="submit" name="login_btn" class="btn btn-success btn-user btn-block"> Login </button>
                                        <hr>
                                    <div class="text-center">
                                        <h1 class="h5 text-gray-900 mb-4" style="font-style: italic; font-size: 25px; font-family: monospace;">"Ibalik and Diwa ng Lawa"</h1>
                                    </div>
                                    </form>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>

            </div>

        </div>

    </div>

    </body>
</html>



